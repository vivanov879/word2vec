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

phrases = read_words('dictionary')
dictionary = {}

for _, sentence in pairs(phrases) do
  
  short_sentence = {}
  for i, word in pairs(sentence) do
    if i ~= #sentence then 
      short_sentence[#short_sentence + 1] = word
    end
  end
  dictionary[tonumber(sentence[#sentence])] = short_sentence
  
end


fd = io.open('dictionary_sorted_by_index', 'w')
for _, sentence in pairs(dictionary) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end
