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

visualize_words = {"the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"}
visualize_indexes = {}
for _, i in pairs(visualize_words) do 
  
  visualize_indexes[#visualize_indexes + 1] = inv_vocabulary_en[visualize_words[i]]
  
  
end

m = torch.load('model')

word_center = torch.Tensor(#vocabulary_en)
for i = 1, #vocabulary_en do 
  word_center[i] = i
end
word_outer = word_center:clone()

_, x_outer, x_center = unpack(m:forward({word_center, word_outer}))

x = x_outer + x_center

mean = x:mean(1)
std = x:std(1)

mean_expanded = torch.expand(mean, x:size(1), x:size(2))
std_expanded = torch.expand(std, x:size(1), x:size(2))

x = x:add(-mean_expanded)
x = x:cdiv(std_expanded)

covariance = torch.mm(x:t(), x)

u,s,v = torch.svd(covariance)




a = 1




