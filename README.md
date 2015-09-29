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
loss = 0.11914144, grad_params:norm() = 2.9512e-03, params:norm() = 9.8262e+01	
loss = 0.11778496, grad_params:norm() = 3.2691e-03, params:norm() = 9.8274e+01	
loss = 0.11646408, grad_params:norm() = 2.9714e-03, params:norm() = 9.8286e+01	
loss = 0.11348016, grad_params:norm() = 2.8652e-03, params:norm() = 9.8298e+01	
loss = 0.11632667, grad_params:norm() = 3.1923e-03, params:norm() = 9.8311e+01	
loss = 0.11513715, grad_params:norm() = 3.1598e-03, params:norm() = 9.8323e+01	
loss = 0.11461787, grad_params:norm() = 2.8214e-03, params:norm() = 9.8335e+01	
loss = 0.11431365, grad_params:norm() = 3.1080e-03, params:norm() = 9.8347e+01	
loss = 0.12215819, grad_params:norm() = 4.3113e-03, params:norm() = 9.8360e+01	
loss = 0.11354396, grad_params:norm() = 3.1466e-03, params:norm() = 9.8372e+01	
loss = 0.11717846, grad_params:norm() = 3.3619e-03, params:norm() = 9.8385e+01	
loss = 0.11574432, grad_params:norm() = 2.9117e-03, params:norm() = 9.8397e+01	
loss = 0.11931591, grad_params:norm() = 3.1105e-03, params:norm() = 9.8410e+01	
loss = 0.11486262, grad_params:norm() = 2.3376e-03, params:norm() = 9.8422e+01	
loss = 0.11434479, grad_params:norm() = 2.6629e-03, params:norm() = 9.8435e+01	
loss = 0.11297270, grad_params:norm() = 3.0902e-03, params:norm() = 9.8447e+01	
```

```th sentiment_analysis.lua``` output after some number of iterations:
```
train set: loss = 1.55104364, f1_score = 0.18394721, precision = 0.34597694, recall = 0.22747006, grad_params:norm() = 6.7158e-01, params:norm() = 6.3476e+01	
dev set:   loss = 1.56834475, f1_score = 0.16801552, precision = 0.30563835, recall = 0.21562503	
success	
train set: loss = 1.55557282, f1_score =    nan, precision = 0.29000712, recall = 0.23968426, grad_params:norm() = 8.7115e-01, params:norm() = 6.3757e+01	
dev set:   loss = 1.56860018, f1_score =    nan, precision = 0.24876680, recall = 0.21427333	
train set: loss = 1.54965129, f1_score = 0.17010089, precision = 0.35540269, recall = 0.21828832, grad_params:norm() = 2.2832e-01, params:norm() = 6.4020e+01	
dev set:   loss = 1.56864349, f1_score = 0.16472635, precision = 0.28914897, recall = 0.21383932	
train set: loss = 1.55040840, f1_score = 0.18304011, precision = 0.35890092, recall = 0.22776055, grad_params:norm() = 7.1540e-01, params:norm() = 6.4286e+01	
dev set:   loss = 1.56893610, f1_score = 0.17009003, precision = 0.30058033, recall = 0.21598218	
success	
train set: loss = 1.55578923, f1_score =    nan, precision = 0.27823843, recall = 0.23775610, grad_params:norm() = 8.7676e-01, params:norm() = 6.4530e+01	
dev set:   loss = 1.56943533, f1_score =    nan, precision = 0.24482958, recall = 0.21237577	
train set: loss = 1.54900364, f1_score = 0.17349076, precision = 0.36970442, recall = 0.22035081, grad_params:norm() = 2.1700e-01, params:norm() = 6.4779e+01	
dev set:   loss = 1.56876765, f1_score = 0.16603706, precision = 0.28586120, recall = 0.21404013	
train set: loss = 1.55048328, f1_score = 0.18201624, precision = 0.35189609, recall = 0.22688347, grad_params:norm() = 7.2782e-01, params:norm() = 6.5017e+01	
dev set:   loss = 1.56869145, f1_score = 0.16786719, precision = 0.30135164, recall = 0.21560279	
```
```th visualize_word_vectors.lua``` output 
![alt tag](https://github.com/vivanov879/word2vec/blob/master/Result_example.png)
