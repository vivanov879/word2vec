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


phrases_filtered_tensor, sentiment_lables_filtered_tensor, phrases_filtered_text = unpack(torch.load('sentiment_features_and_labels'))

assert (phrases_filtered_tensor:size(1) == sentiment_lables_filtered_tensor:size(1))
assert (phrases_filtered_tensor:size(1) == #phrases_filtered_text)

batch_size = 3000
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
x = nn.Linear(phrases_filtered_tensor:size(2), 200)(x_raw)
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

  local _, loss = optim.adam(feval, params, optim_state)
  if i % 100 == 0 then
    
    local loss_train = loss[1]
    local _, predicted_class  = prediction:max(2)
    local f1_score_train, precision_train, recall_train = unpack(calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2))))
    print(string.format("train set: loss = %6.8f, f1_score = %6.8f, precision = %6.8f, recall = %6.8f, grad_params:norm() = %6.4e", loss_train, f1_score_train, precision_train, recall_train, grad_params:norm()))

    
  end
  
end







pass_dummy = 1