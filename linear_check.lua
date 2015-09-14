require 'mobdebug'.start()
require 'nn'
local model_utils=require 'model_utils'
require 'nngraph'
nngraph.setDebug(true)
require 'linear1'

local m = nn.Linear1(2, 3)

local jac = nn.Jacobian

local precision = 1e-3

local input = torch.rand(4,2)

local err = jac.testJacobian(m,input)
assert(err < precision)


local params, grad_params = m:getParameters()

local err = jac.testJacobianParameters(m, input, params, grad_params)
print(err)
assert(err < precision)


