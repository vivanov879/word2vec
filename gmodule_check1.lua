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
m_half = nn.gModule({x_raw}, {x})

x_raw1 = nn.Identity()()
x_raw2 = nn.Identity()()
x1 = m_half({x_raw1})
x2 = m_half({x_raw2})

x1 = nn.MulConstant(-1)(x1)
d = nn.CAddTable()({x1, x2})
d = nn.Power(2)(d)
d = nn.Linear(10,1)(d)
m = nn.gModule({x_raw1, x_raw2}, {d})


x = nn.Identity()()
z = m_half({x})
m = nn.gModule({x}, {z})

local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)

