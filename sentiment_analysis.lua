require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)



function calc_f1(prediction, target)
  local f1_accum = 0
  local precision_accum = 0
  local recall_accum = 0
  for c = 1, 5 do
    local p = torch.eq(prediction, c):double()
    local t = torch.eq(target, c):double()
    local true_positives = torch.mm(t:t(),p)[1][1]
        
    p = torch.eq(prediction, c):double()
    t = torch.ne(target, c):double()
    local false_positives = torch.mm(t:t(),p)[1][1]
    
    p = torch.ne(prediction, c):double()
    t = torch.eq(target, c):double()
    local false_negatives = torch.mm(t:t(),p)[1][1]
    
    local precision = true_positives / (true_positives + false_positives)
    local recall = true_positives / (true_positives + false_negatives)
    
    local f1_score = 2 * precision * recall / (precision + recall)
    f1_accum = f1_accum + f1_score 
    precision_accum = precision_accum + precision
    recall_accum = recall_accum + recall
    
    
  end
  return {f1_accum / 5, precision_accum / 5, recall_accum / 5}
end


inv_vocabulary_en = table.load('inv_vocabulary_en')
vocabulary_en = table.load('vocabulary_en')

features_train, labels_train, text_train = unpack(torch.load('sentiment_train.t7'))
phrases_dev, labels_dev, text_dev= unpack(torch.load('sentiment_dev.t7'))


assert (features_train:size(1) == labels_train:size(1))
assert (features_train:size(1) == #text_train)

batch_size = 3000
data_index = 1
n_data = features_train:size(1)

function gen_batch()
  start_index = data_index
  end_index = math.min(n_data, start_index + batch_size - 1)
  if end_index == n_data then
    data_index = 1
  else
    data_index = data_index + batch_size
  end
    
  features = features_train[{{start_index, end_index}, {}}]
  labels = labels_train[{{start_index, end_index}}]
  
  text = text_train[start_index]
  text_readable = {}
  for i, word in pairs(text) do 
    text_readable[#text_readable + 1] = vocabulary_en[word]
  end
  text_readable = table.concat(text_readable, ' ')
 
  return features, labels, text_readable

end


x_raw = nn.Identity()()
x = nn.Linear(features_train:size(2), 200)(x_raw)
x = nn.Tanh()(x)
x = nn.Linear(200, 5)(x)
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
    
    features, labels, text_readable = gen_batch()
            
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

  local _, loss = optim.adam(feval, params, optim_state)
  if i % 100 == 0 then
    
    local loss_train = loss[1]
    local _, predicted_class  = prediction:max(2)
    local f1_score_train, precision_train, recall_train = unpack(calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2))))
    
    local features = features_dev[{{}, {}}]
    local labels = labels_dev[{{}}]
    local prediction, h = unpack(m:forward(features))
    local _, predicted_class  = prediction:max(2)
    local loss_dev = criterion:forward(prediction, labels)
    local f1_score_dev, precision_dev, recall_dev = unpack(calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2))))
    
    print(string.format("train set: loss = %6.8f, f1_score = %6.8f, precision = %6.8f, recall = %6.8f, grad_params:norm() = %6.4e", loss_train, f1_score_train, precision_train, recall_train, grad_params:norm()))
    print(string.format("dev set:   loss = %6.8f, f1_score = %6.8f, precision = %6.8f, recall = %6.8f", loss_dev, f1_score_dev, precision_dev, recall_dev))

    
  end
  
end







pass_dummy = 1