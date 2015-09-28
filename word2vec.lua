require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


function get_context_words(sentence, context_size, center_word_index)
  local possible= {}
  for i = -context_size, context_size do 
    if i ~= 0 and i + center_word_index <= #sentence and i + center_word_index > 0 then
      possible[#possible+ 1] = sentence[i + center_word_index]
    end
  end
  local ids = {}
  for k = 1, 2*context_size do
    ids[#ids + 1] = possible[math.random(1, #possible)]
  end
  return ids
end


function read_words(fn)
  fd = io.lines(fn)
  sentences = {}
  line = fd()

  while line do
    sentence = {}
    for _, word in pairs(string.split(line, " ")) do
        sentence[#sentence + 1] = word
    end
    sentences[#sentences + 1] = sentence
    line = fd()
  end
  return sentences
end

function math.clamp(x, min_val, max_val)
  if x < min_val then
    x = min_val
  elseif x > max_val then
    x = max_val
  end
  return x
end


function convert2tensors(sentences)
  l = {}
  for _, sentence in pairs(sentences) do
    t = torch.zeros(1, #sentence)
    for i = 1, #sentence do 
      t[1][i] = sentence[i]
    end
    l[#l + 1] = t
  end
  return l  
end


sentences, vocabulary, inv_vocabulary = unpack(torch.load('filter_sentences_output.t7'))

n_data = #sentences
vocab_size = #vocabulary


function calc_max_sentence_len(sentences)
  local m = 1
  for _, sentence in pairs(sentences) do
    m = math.max(m, #sentence)
  end
  return m
end

max_sentence_len = calc_max_sentence_len(sentences)
context_size = 5
batch_size = 1000
neg_samples_num = 10

data_index = 1

function gen_batch()
  start_index = data_index
  end_index = math.min(n_data, start_index + batch_size - 1)
  if end_index == n_data then
    data_index = 1
  else
    data_index = data_index + batch_size
  end
  basic_batch_size = end_index - start_index + 1
  local center_words = torch.Tensor( (2*context_size * (1 + neg_samples_num)) * basic_batch_size)
  local outer_words = torch.Tensor( (2*context_size * (1 + neg_samples_num)) * basic_batch_size)
  local labels = torch.Tensor( center_words:size(1))
  row = 1
  for k = 1, basic_batch_size do    
    sentence = sentences[start_index + k - 1]
    center_word_index = math.random(1, #sentence)
    center_word = sentence[center_word_index]
    context_words = get_context_words(sentence, context_size, center_word_index)
    for _, outer_word in pairs(context_words) do
        center_words[row] = center_word
        outer_words[row] = outer_word
        labels[row] = 1
        row = row + 1
        neg_samples = torch.rand(neg_samples_num):mul(vocab_size):floor():add(1)
        outer_words[{{row, row+neg_samples_num-1}}] = neg_samples
        center_words[{{row, row+neg_samples_num-1}}]:fill(center_word)
        labels[{{row, row+neg_samples_num-1}}] = torch.Tensor(neg_samples_num):fill(-1)
        row = row + neg_samples_num
        dummy_pass = 1
    end
  end
  return center_words, outer_words, labels
end

word_center = nn.Identity()()
word_outer = nn.Identity()()

x_center = Embedding(vocab_size, 10)(word_center)
x_center = nn.Normalize(2)(x_center)
x_outer = Embedding(vocab_size, 10)(word_outer)
x_outer = nn.Normalize(2)(x_outer)

z = nn.CosineDistance()({x_outer, x_center})

m = nn.gModule({word_center, word_outer}, {z, x_outer, x_center})

local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)

criterion = nn.MarginCriterion()

function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    local loss = 0
    
    center_words, outer_words, labels = gen_batch()
    
    ------------------- forward pass -------------------
    z, x_outer, x_center = unpack(m:forward({center_words, outer_words}))
    loss_m = criterion:forward(z, labels)
    loss = loss + loss_m
    
    -- complete reverse order of the above
    dx_outer = torch.zeros(x_outer:size())
    dx_center = torch.zeros(x_center:size())
    dz = criterion:backward(z, labels)
    dcenter_words, douter_words = unpack(m:backward({center_words, outer_words}, {dz, dx_outer, dx_center}))
    

    return loss, grad_params

end



optim_state = {learningRate = 1e-2}


for i = 1, 1000000 do

  local _, loss = optim.adam(feval, params, optim_state)
  if i % 1 == 0 then
    print(string.format( 'loss = %6.8f, grad_params:norm() = %6.4e, params:norm() = %6.4e', loss[1], grad_params:norm(), params:norm()))
  end
  
  if i % 10 == 0 then
    torch.save('model.t7', m)
  end
  
end

