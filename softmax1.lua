require 'mobdebug'.start()
require 'nn'
local model_utils=require 'model_utils'
require 'nngraph'
nngraph.setDebug(true)
local SoftMax1 = torch.class('nn.SoftMax1', 'nn.Module')

function SoftMax1:updateOutput(input)
  
  self.output:resizeAs(input):copy(input)
  local e = torch.exp(input)
  local s = torch.sum(e, 2)
  s = torch.expand(s, input:size(1), input:size(2))
  self.output = torch.cdiv(e, s)
  
  return self.output
end

function SoftMax1:updateGradInput(input, gradOutput)
  
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  for i = 1, input:size(1) do
    local jacobian = torch.zeros(input:size(2), input:size(2))
    for p = 1, input:size(2) do
      for q = 1, input:size(2) do
        if p == q then
          jacobian[p][q] = self.output[i][p] * (1 - self.output[i][q])
        else
          jacobian[p][q] = self.output[i][p] * (  - self.output[i][q])
        end
      end
      
    end
    self.gradInput[{{i}, {}}] = torch.mm(gradOutput[{{i}, {}}], jacobian)
  end

  return self.gradInput
end

