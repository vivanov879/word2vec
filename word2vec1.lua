require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


--train data
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


sentences_en = read_words('filtered_datasetSentences_indexes_en')

n_data = #sentences_en

vocabulary_en = table.load('vocabulary_en')
vocab_size = #vocabulary_en


function calc_max_sentence_len(sentences)
  local m = 1
  for _, sentence in pairs(sentences_en) do
    m = math.max(m, #sentence)
  end
  return m
end

max_sentence_len = calc_max_sentence_len(sentences_en)
context_size = 6
batch_size = 5
data_index = 1

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1

  end
  start_index = end_index - batch_size

  sentences = sentences_en
  
  local batch = torch.zeros(batch_size, 3)
  for k = 1, batch_size do
    sentence = sentences[start_index + k - 1]
    center_word_index = math.random(#sentence)
    center_word = sentence[center_word_index]
    context_index = center_word_index + (math.random() > 0.5 and 1 or -1) * math.random(2, math.floor(context_size/2))
    context_index = math.clamp(context_index, 1, #sentence)
    outer_word = sentence[context_index]
    neg_word = math.random(#vocabulary_en)
    batch[k][1] = center_word
    batch[k][2] = outer_word
    batch[k][3] = neg_word
    
  end
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  return batch
end



x_raw = nn.Identity()()
x = nn.Linear(12, 5)(x_raw)
x = nn.Tanh()(x)
x = nn.Linear(5, 10)(x)
m1 = nn.gModule({x_raw}, {x})

m1_clones = model_utils.clone_many_times(m1, 2)

x_raw1 = nn.Identity()()
x_raw2 = nn.Identity()()
x1 = m1_clones[1]({x_raw1})
x2 = m1_clones[2]({x_raw2})

x1 = nn.MulConstant(-1)(x1)
d = nn.CAddTable()({x1, x2})
d = nn.Power(2)(d)
d = nn.Linear(10,1)(d)
m2 = nn.gModule({x_raw1, x_raw2}, {d})

m2_clones = model_utils.clone_many_times(m2, 2)

x_center = nn.Identity()()
x_outer = nn.Identity()()
x_neg = nn.Identity()()
d_outer = m2_clones[1]({x_center, x_outer})
d_neg = m2_clones[2]({x_center, x_neg})
target_outer = nn.Identity()()
target_neg = nn.Identity()()
loss1 = nn.MarginCriterion()({d_outer, target_outer})
loss2 = nn.MarginCriterion()({d_neg, target_neg})
loss_m = nn.CAddTable()({loss1, loss2})
m = nn.gModule({target_outer, target_neg, x_center, x_outer, x_neg}, {loss_m})


embed_center = Embedding(vocab_size, 12)
embed_outer = Embedding(vocab_size, 12)

local params, grad_params = model_utils.combine_all_parameters(m, embed_center, embed_outer)
params:uniform(-0.08, 0.08)

function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    local loss = 0
    
    batch = gen_batch()
    word_center = batch[{{},1}]
    word_outer = batch[{{},2}]
    word_neg = batch[{{},3}]
        
    ------------------- forward pass -------------------
    x_center = embed_center:forward(word_center)
    x_outer = embed_outer:forward(word_outer)
    x_neg = embed_outer:forward(word_neg)
    
    target_outer = torch.Tensor(x_outer:size(1), 1):fill(1)
    target_neg = torch.Tensor(x_neg:size(1), 1):fill(-1)
    
    loss_m = m:forward({target_outer, target_neg, x_center, x_outer, x_neg})
    loss = loss + loss_m[1]
    
    
    -- complete reverse order of the above
    dloss_m = torch.ones(loss_m:size())
    dtarget_outer, dtarget_neg, dx_center, dx_outer, dx_neg = unpack(m:backward({target_outer, target_neg, x_center, x_outer, x_neg}, dloss_m))
    dword_center = embed_center:backward(word_center, dx_center)
    dword_outer = embed_center:backward(word_outer, dx_outer)
    dword_neg = embed_center:backward(word_neg, dx_neg)
    
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)
    print(grad_params)
    return loss, grad_params

end



optim_state = {learningRate = 1e-5}



for i = 1, 1000 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  if i % 100 == 0 then
    print(loss)
  end

end


