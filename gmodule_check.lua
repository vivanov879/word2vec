require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)





x = nn.Identity()()
z = nn.Linear(12, 5)(x)
z = nn.Tanh()(z)
z = nn.Linear(5, 10)(z)
z = nn.Copy()(x)
m_half = nn.gModule({x}, {z})

x = nn.Identity()()
z = m_half({x})
m = nn.gModule({x}, {z})

local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)
