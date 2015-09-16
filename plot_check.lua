require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)
require 'gnuplot'


x = torch.Tensor(100)
y = torch.Tensor(100)

for i = 1, x:size(1) do 
  x[i] = i
  y[i] = i + math.random()
  
  
end


gnuplot.plot(x, y)

a = 1