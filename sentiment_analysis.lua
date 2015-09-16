require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)

inv_vocabulary_en = table.load('inv_vocabulary_en')
vocabulary_en = table.load('vocabulary_en')

indexes = torch.Tensor(#vocabulary_en)
for i = 1, indexes:size(1) do 
  indexes[i] = i
end

m = torch.load('model')

word_center = indexes:clone()
word_outer = indexes:clone()

_, x_outer, x_center = unpack(m:forward({word_center, word_outer}))

word_vectors = x_outer + x_center

batch_size = 100
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
  local target = 1
  if data_index % 2 == 0 then
    target = -1
  end
  for k = 1, batch_size do
    sentence = sentences[start_index + k - 1]
    center_word_index = math.random(#sentence)
    center_word = sentence[center_word_index]
    context_index = center_word_index + (math.random() > 0.5 and 1 or -1) * math.random(2, math.floor(context_size/2))
    context_index = math.clamp(context_index, 1, #sentence)
    outer_word = sentence[context_index]
    neg_word = math.random(#vocabulary_en)
    batch[k][1] = center_word
    if target == 1 then
      batch[k][2] = outer_word
    else 
      batch[k][2] = neg_word
    end
  end
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  return batch, target
end




a = 1