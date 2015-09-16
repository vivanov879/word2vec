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


dictionary = read_words('dictionary_sorted_by_index')
sentiment_labels_sentences = read_words('sentiment_labels')
sentiment_labels = {}

for i, sentence in pairs(sentiment_labels_sentences) do 
  sentiment_labels[tonumber(sentence[1])] = tonumber(sentence[#sentence])
end

batch_size = 100
data_index = 1

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1
  end
  start_index = end_index - batch_size

  sentences = dictionary
  
  local batch = torch.zeros(batch_size, word_vectors:size(2))
  
  if data_index % 2 == 0 then
    target = -1
  end
  for k = 1, batch_size do
    
    
    sentence = sentences[start_index + k - 1]
    
    
    
  end
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  return batch, target
end




a = 1