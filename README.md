Torch implementation of word2vec and sentiment analysis as explained in assignment1.pdf. Negative sampling loss function and max-margin learning are used for word2vec.

- In Terminal.app, run ```python extract_datasets_for_torch.py``` to generate datasets for sentiment_analysis.lua.
- Run ```th filter_sentences.lua``` to generate vocabulary of words and then write sentences to file using this vocabulary
- Run ```th word2vec.lua``` to train the word2vec model. You do not need a complex multi layer network to generate embeddings here. Our task is to simply separate word embeddings based on cooccurance. If we had documents here, we would need a complex system, for example a convnet for documents, to generate the embeddings, because there is no obvious way to combine multiple documents in a simple embedding. Here we simply have a vocabulary of words and map an embedding to each word. You can try it yourself: add several layers into the network and use the very last layer(the layer before the difference between the embedings is put into z) or some middle layer, or embdedding layer, as an output embedding for the words. Getting embeddings from a single embedding layer is logical, because the model can adjust each value in embedding matrix to pull words together or apart.
- Run ```th visualize_word_vectors.lua``` to project several words into 2d space using SVD and then plot this words. You'll get something like Result_example.png.
- Run ```th gen_sentiment_files.lua``` generates training and dev sets of words and their labels
- Run ```th sentiment_analysis.lua``` trains a sentiment analysis model.

You can use the program with your own text. ``` filter_sentences.lua``` works with tokenized input text. You can tokenize you text using https://github.com/moses-smt/mosesdecoder by running 
```
cd ~/mosesdecoder/scripts/tokenizer/ 
./pre-tokenizer.perl ~/word2vec/trainSentences_raw | ./lowercase.perl | ./tokenizer.perl > ~/word2vec/trainSentences
``` 

Different models in word2vec.lua and their results.
```
word_center = nn.Identity()()
word_outer = nn.Identity()()

x_center_ = Embedding(vocab_size, 100)(word_center)
x_center = nn.Linear(100, 50)(x_center_)
x_center = nn.Tanh()(x_center)

x_outer_ = Embedding(vocab_size, 100)(word_outer)
x_outer = nn.Linear(100, 50)(x_outer_)
x_outer = nn.Tanh()(x_outer)

x_center_minus = nn.MulConstant(-1)(x_center)

z = nn.CAddTable()({x_outer, x_center_minus})
z = nn.Power(2)(z)
z = nn.Sum(2)(z)

m = nn.gModule({word_center, word_outer}, {z, x_outer_, x_center_})
```

```th word2vec.lua``` example outputs after many iterations:
```
loss = 0.11931591, grad_params:norm() = 3.1105e-03, params:norm() = 9.8410e+01	
loss = 0.11486262, grad_params:norm() = 2.3376e-03, params:norm() = 9.8422e+01	
loss = 0.11434479, grad_params:norm() = 2.6629e-03, params:norm() = 9.8435e+01	
loss = 0.11297270, grad_params:norm() = 3.0902e-03, params:norm() = 9.8447e+01
```

```th sentiment_analysis.lua``` output after some number of iterations:
```
train set: loss = 1.54543872, f1_score =    nan, precision = 0.34005939, recall = 0.24525582, grad_params:norm() = 9.1239e-01, params:norm() = 1.7219e+02	
dev set:   loss = 1.57993113, f1_score = 0.15993812, precision = 0.30514166, recall = 0.20978553	
success	
train set: loss = 1.53537633, f1_score = 0.18173430, precision = 0.42847230, recall = 0.22450341, grad_params:norm() = 2.6352e-01, params:norm() = 1.7223e+02	
dev set:   loss = 1.58010039, f1_score = 0.15950854, precision = 0.34178091, recall = 0.20911573	
success	
train set: loss = 1.52821191, f1_score = 0.21457752, precision = 0.43319328, recall = 0.24664033, grad_params:norm() = 6.9045e-01, params:norm() = 1.7227e+02	
dev set:   loss = 1.58173595, f1_score =    nan, precision = 0.22474973, recall = 0.20582198	
train set: loss = 1.54566442, f1_score =    nan, precision = 0.32999437, recall = 0.24216919, grad_params:norm() = 9.0763e-01, params:norm() = 1.7229e+02	
dev set:   loss = 1.58140066, f1_score =    nan, precision = 0.23484737, recall = 0.20707198	
```
