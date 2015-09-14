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

sentences_en = read_words('filtered_sentences_indexes_en1')

function calc_max_sentence_len(sentences)
  local m = 1
  for _, sentence in pairs(sentences_en) do
    m = math.max(m, #sentence)
  end
  return m
end

max_sentence_len = math.max(calc_max_sentence_len(sentences_en), calc_max_sentence_len(sentences_ru))

--sentences_ru = convert2tensors(sentences_ru)
--sentences_en = convert2tensors(sentences_en)

--print(sentences_ru)

assert(#sentences_en == #sentences_ru)
n_data = #sentences_en

vocabulary_ru = table.load('vocabulary_ru')
vocabulary_en = table.load('vocabulary_en')
vocab_size = #vocabulary_ru
assert (#vocabulary_en == #vocabulary_ru)


