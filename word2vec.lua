require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


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


sentences, vocabulary, inv_vocabulary = unpack(torch.load('filter_sentences_output'))

n_data = #sentences

vocabulary = table.load('vocabulary_en')
inv_vocabulary = table.load('inv_vocabulary_en')
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
neg_samples_num = 7

n_data = batch_size * math.floor(n_data/batch_size)
data_index = 1

function gen_batch()
  start_index = data_index
  end_index = math.min(n_data, start_index + batch_size - 1)
  if end_index == n_data then
    data_index = 1
  else
    data_index = data_index + batch_size
  end
  basic_batch_size = batch_size
  local center_words = torch.Tensor( (2*context_size * (1 + neg_samples_num)) * basic_batch_size)
  local outer_words = torch.Tensor( (2*context_size * (1 + neg_samples_num)) * basic_batch_size)
  local labels = torch.Tensor( center_words:size(1))
  row = 1
  for k = 1, basic_batch_size do    
    sentence = sentences[start_index + k - 1]
    center_word_index = math.random(2, #sentence-1)
    center_word = sentence[center_word_index]
    for i = -context_size, context_size do
      if i ~= 0 then 
        context_index = center_word_index + i
        context_index = math.clamp(context_index, 1, #sentence)
        outer_word = sentence[context_index]
        center_words[row] = center_word
        outer_words[row] = outer_word
        labels[row] = 1
        row = row + 1
        neg_samples = torch.rand(neg_samples_num):mul(vocab_size):byte():double():add(1)
        outer_words[{{row, row+neg_samples_num-1}}] = neg_samples
        center_words[{{row, row+neg_samples_num-1}}]:fill(center_word)
        labels[{{row, row+neg_samples_num-1}}] = torch.Tensor(neg_samples_num):fill(-1)
        row = row + neg_samples_num
        dummy_pass = 1
      end
    end
  end
  return center_words, outer_words, labels
end

word_center = nn.Identity()()
word_outer = nn.Identity()()

x_center = Embedding(vocab_size, 12)(word_center)
x_outer = Embedding(vocab_size, 12)(word_outer)

x_center = nn.Linear(12, 5)(x_center)
x_center = nn.Tanh()(x_center)
x_center = nn.Linear(5, 10)(x_center)

x_outer = nn.Linear(12, 5)(x_outer)
x_outer= nn.Tanh()(x_outer)
x_outer = nn.Linear(5, 10)(x_outer)

x_center_minus = nn.MulConstant(-1)(x_center)

z = nn.CAddTable()({x_outer, x_center_minus})
z = nn.Power(2)(z)
z = nn.Sum(2)(z)

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
    
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)
    return loss, grad_params

end



optim_state = {learningRate = 1e-1}


for i = 1, 1000000 do

  local _, loss = optim.adam(feval, params, optim_state)
  if i % 1 == 0 then
    print(string.format( 'loss = %6.8f', loss[1]))
    
  end
  
  if i % 10 == 0 then
    torch.save('model', m)
  end
  
end

