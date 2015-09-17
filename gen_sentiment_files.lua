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


phrases = read_words('trainSentences')
sentiment_lables = read_words('trainLabels')
assert(#phrases == #sentiment_lables)
phrases_filtered = {}
phrases_filtered_text = {}
sentiment_lables_filtered = {}

for index_phrases, sentence in pairs(phrases) do
  local short_sentence = {}
  for i, word in pairs(sentence) do
    if inv_vocabulary_en[word] ~= nil then 
      short_sentence[#short_sentence + 1] = inv_vocabulary_en[word]
    end
  end
  if #short_sentence > 0 then
    local t = torch.Tensor(#short_sentence, word_vectors:size(2))
    for i, word in pairs(short_sentence) do 
      t[{{i}, {}}] = word_vectors[{{short_sentence[i]}, {}}]
    end
    phrases_filtered[#phrases_filtered + 1] = t:mean(1)
    phrases_filtered_text[#phrases_filtered_text + 1] = short_sentence

    sentiment_labels_sentence = sentiment_lables[index_phrases]
    sentiment_lables_filtered[#sentiment_lables_filtered + 1] = tonumber(sentiment_labels_sentence[1])
    
    assert(#sentiment_lables_filtered == #phrases_filtered)
    
  end
end


phrases_filtered_tensor = torch.Tensor(#phrases_filtered, word_vectors:size(2))
sentiment_lables_filtered_tensor = torch.Tensor(#phrases_filtered, 1)
for i, _ in pairs(phrases_filtered) do 
  phrases_filtered_tensor[{{i}, {}}] = phrases_filtered[i]
  sentiment_lables_filtered_tensor[{{i}, {}}] = sentiment_lables_filtered[i]
end

torch.save('sentiment_features_and_labels', {phrases_filtered_tensor, sentiment_lables_filtered_tensor, phrases_filtered_text})


