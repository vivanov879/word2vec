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


sentences_en = read_words('filtered_datasetSentences_indexes_en')

n_data = #sentences_en

vocabulary_en = table.load('vocabulary_en')
vocab_size = #vocabulary_en






x_raw = nn.Identity()()
x = nn.Linear(12, 5)(x_raw)
x = nn.Tanh()(x)
x = nn.Linear(5, 10)(x)
m_half = nn.gModule({x_raw}, {x})

x_raw1 = nn.Identity()()
x_raw2 = nn.Identity()()
x1 = m_half(x_raw1)
x2 = m_half(x_raw2)

x1 = nn.MulConstant(-1)(x1)
d = nn.CAddTable()({x1, x2})
d = nn.Power(2)(d)
d = nn.Linear(10,1)(d)
m = nn.gModule({x_raw1, x_raw2}, {d})

criterion=nn.MarginCriterion(1)

embed = Embedding(vocab_size, 12)









