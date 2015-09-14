require 'mobdebug'.start()
require 'nn'
local model_utils=require 'model_utils'
require 'nngraph'
nngraph.setDebug(true)
require 'softmax1'



local x = nn.Identity()()
local y = nn.SoftMax1()(x)
local m = nn.gModule({x},{y})

local mytester = torch.Tester()


local jac = nn.Jacobian

local precision = 1e-3

local input = torch.rand(10,5)

local err = jac.testJacobian(m,input)
assert(err < precision)

