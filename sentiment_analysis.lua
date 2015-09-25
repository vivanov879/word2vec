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

batch_size = 10
data_index = 1
n_data = phrases_filtered_tensor:size(1)

function gen_batch()
  start_index = data_index
  end_index = math.min(n_data, start_index + batch_size - 1)
  if end_index == n_data then
    data_index = 1
  else
    data_index = data_index + batch_size
  end
    
  features = phrases_filtered_tensor[{{start_index, end_index}, {}}]
  labels = sentiment_lables_filtered_tensor[{{start_index, end_index}}]
  
  text_first_sentence = phrases_filtered_text[start_index]
  text_first_sentence_readable = {}
  for i, word in pairs(text_first_sentence) do 
    text_first_sentence_readable[#text_first_sentence_readable + 1] = vocabulary_en[word]
  end
  text_first_sentence_readable = table.concat(text_first_sentence_readable, ' ')
 
  return features, labels, text_first_sentence_readable

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
    
    features, labels, text_first_sentence_readable = gen_batch()
            
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
    print(text_first_sentence_readable)
    local _, predicted_class  = prediction:max(2)
    print(prediction[{1, {}}])
    print(string.format("predicted class = %d, target class = %d, loss = %6.8f, gradnorm = %6.4e", predicted_class[1][1], labels[1], loss[1], grad_params:norm()))
    print(params:clone():pow(2):sum())
    
  end
  
end







pass_dummy = 1