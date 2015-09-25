require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)
require 'gnuplot'

inv_vocabulary_en = table.load('inv_vocabulary_en')
vocabulary_en = table.load('vocabulary_en')

visualize_words = {"the", "a", "an", ",", ".", "?", "!",  "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying", "movie", "ordinary"}
visualize_indexes = torch.Tensor(#visualize_words)
for i, _ in pairs(visualize_words) do 
  visualize_indexes[i] = inv_vocabulary_en[visualize_words[i]]
end

m = torch.load('model.t7')

word_center = visualize_indexes:clone()
word_outer = visualize_indexes:clone()

_, x_outer, x_center = unpack(m:forward({word_center, word_outer}))

x = torch.add(x_outer, x_center)

mean = x:mean(1)
std = x:std(1)

mean_expanded = torch.expand(mean, x:size(1), x:size(2))
std_expanded = torch.expand(std, x:size(1), x:size(2))

x = x:add(-mean_expanded)
x = x:cdiv(std_expanded)

covariance = torch.mm(x:t(), x)

u,s,v = torch.svd(covariance)

u_reduced = u[{{}, {1, 2}}]
x_projected = torch.mm(x, u_reduced)
print(x_projected)
gnuplot.plot(x_projected, '.')
for i = 1, visualize_indexes:size(1) do 
  --gnuplot.raw(" set label 'ward' at ( 0.12, 0.54 ) ")
  gnuplot.raw(" set label '" .. visualize_words[i] .. "' at " .. x_projected[i][1] .. "," .. x_projected[i][2] .. " " )
end
a = 1




