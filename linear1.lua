require 'mobdebug'.start()
require 'nn'
local model_utils=require 'model_utils'
require 'nngraph'
nngraph.setDebug(true)

local Linear1, parent = torch.class('nn.Linear1', 'nn.Module')

function Linear1:__init(inputSize, outputSize)
  parent.__init(self)
  self.weight = torch.Tensor(outputSize, inputSize)
  self.bias = torch.Tensor(1, outputSize)
  self.gradWeight = torch.Tensor(outputSize, inputSize)
  self.gradBias = torch.Tensor(1, outputSize)
  
end

function Linear1:updateOutput(input)
  
  self.output:resizeAs(input):copy(input)
  self.output = torch.mm(input, self.weight:t())
  local bias_expanded = torch.expand(self.bias, self.output:size(1), self.output:size(2))
  self.output = torch.add(self.output, bias_expanded)
   
  return self.output
end

function Linear1:updateGradInput(input, gradOutput)
  
  self.gradInput:resizeAs(input)
  self.gradInput = torch.mm(gradOutput, self.weight)

  return self.gradInput
end


function Linear1:accGradParameters(input, gradOutput)
  self.gradWeight:addmm(gradOutput:t(), input)
  self.gradBias:add(torch.sum(gradOutput, 1))
  --self.gradBias:mul(10000)

end


Linear1.sharedAccUpdateGradParameters = Linear1.accUpdateGradParameters
