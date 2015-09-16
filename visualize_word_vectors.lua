require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


inv_vocabulary_en = table.load('inv_vocabulary_en')


visualize_words = {"the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"}
visualize_indexes = {}
for _, i in pairs(visualize_words) do 
  
  visualize_indexes[#visualize_indexes + 1] = inv_vocabulary_en[visualize_words[i]]
  
  
end



