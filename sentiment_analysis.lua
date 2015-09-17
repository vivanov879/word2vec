require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)

phrases_filtered_tensor, sentiment_lables_filtered_tensor, phrases_filtered_text = unpack(torch.load('sentiment_features_and_labels'))

batch_size = 10000
data_index = 1
n_data = phrases_filtered_tensor:size(1)

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1
  end
  start_index = end_index - batch_size
  
  features = phrases_filtered_tensor[{{data_index, data_index + batch_size}, {}}]
  labels = sentiment_lables_filtered_tensor[{{data_index, data_index + batch_size}, {}}]
      
  data_index = data_index + 1
  
  return features, labels
end


x_raw = nn.Identity()()
x = nn.Linear(phrases_filtered_tensor:size(2), 5)(x_raw)
x = nn.Tanh()(x)
x = nn.Linear(5, 1)(x)
x = nn.Sigmoid()(x)
m = nn.gModule({x_raw}, {x})


local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)


criterion = nn.MSECriterion()


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
    
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)
    
    return loss, grad_params

end




optim_state = {learningRate = 1e-5}


for i = 1, 1000000 do

  local _, loss = optim.adagrad(feval, params, optim_state)
  if i % 1000 == 0 then
    print(loss)
  end
  
end







pass_dummy = 1