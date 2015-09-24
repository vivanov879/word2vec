require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


inv_vocabulary_en = table.load('inv_vocabulary_en')
vocabulary_en = table.load('vocabulary_en')


phrases_filtered_tensor, sentiment_lables_filtered_tensor, phrases_filtered_text = unpack(torch.load('sentiment_features_and_labels'))

assert (phrases_filtered_tensor:size(1) == sentiment_lables_filtered_tensor:size(1))
assert (phrases_filtered_tensor:size(1) == #phrases_filtered_text)

batch_size = 1000
data_index = 1
n_data = phrases_filtered_tensor:size(1)

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1
  end
  start_index = end_index - batch_size
  data_index = data_index + batch_size
  
  features = phrases_filtered_tensor[{{start_index, end_index - 1}, {}}]
  labels = sentiment_lables_filtered_tensor[{{start_index, end_index - 1}}]
  return features, labels
end


x_raw = nn.Identity()()
x = nn.Linear(phrases_filtered_tensor:size(2), 20)(x_raw)
x = nn.Tanh()(x)
x = nn.Linear(20, 5)(x)
x = nn.LogSoftMax()(x)
m = nn.gModule({x_raw}, {x})


local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)


criterion = nn.ClassNLLCriterion()


function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    local loss = 0
    
    features, labels = gen_batch()
            
    ------------------- forward pass -------------------
    prediction = m:forward(features)
    loss_m = criterion:forward(prediction, labels)
    loss = loss + loss_m
    
    -- complete reverse order of the above
    dprediction = criterion:backward(prediction, labels)
    dfeatures = m:backward(features, dprediction)
    
    return loss, grad_params

end




optim_state = {learningRate = 1e-1}


for i = 1, 1000000 do

  local _, loss = optim.adagrad(feval, params, optim_state)
  if i % 100 == 0 then
    print(string.format("loss = %6.8f, gradnorm = %6.4e", loss[1], grad_params:norm()))
    
  end
  
end







pass_dummy = 1