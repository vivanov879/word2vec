require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   print(pred)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

x = nn.Identity()()
y = nn.Identity()()
xx = nn.Linear(10, 5)(x)
yy = nn.Linear(10, 5)(y)
h = nn.MulConstant(-1)(yy)
d = nn.CAddTable()({xx, h})
d = nn.Power(2)(d)
d = nn.Linear(5,1)(d)
m = nn.gModule({x, y}, {d})

x1 = torch.rand(10)
x2 = torch.rand(10)
criterion=nn.MarginCriterion(1)

for i = 1, 30 do
   gradUpdate(m, {x1, x2}, torch.Tensor(1):fill(-1), criterion, 0.01)
   
end

