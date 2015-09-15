require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)



x_raw = nn.Identity()()
x = nn.Linear(12, 5)(x_raw)
x = nn.Tanh()(x)
x = nn.Linear(5, 10)(x)
m1 = nn.gModule({x_raw}, {x})

m1_clones = model_utils.clone_many_times(m1, 2)

x_raw1 = nn.Identity()()
x_raw2 = nn.Identity()()
x1 = m1_clones[1]({x_raw1})
x2 = m1_clones[2]({x_raw2})

x1 = nn.MulConstant(-1)(x1)
d = nn.CAddTable()({x1, x2})
d = nn.Power(2)(d)
d = nn.Linear(10,1)(d)
m2 = nn.gModule({x_raw1, x_raw2}, {d})

m2_clones = model_utils.clone_many_times(m2, 2)

x_center = nn.Identity()()
x_outer = nn.Identity()()
x_neg = nn.Identity()()
d_outer = m2_clones[1]({x_center, x_outer})
d_neg = m2_clones[2]({x_center, x_neg})
target_outer = nn.Identity()()
target_neg = nn.Identity()()
loss1 = nn.MarginCriterion()({d_outer, target_outer})
loss2 = nn.MarginCriterion()({d_neg, target_neg})
loss_m = nn.CAddTable()({loss1, loss2})
m = nn.gModule({target_outer, target_neg, x_center, x_outer, x_neg}, {loss_m})



local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)
