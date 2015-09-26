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

```th sentiment_analysis.lua``` output after some number of iterations:
```
[0mtrain set: loss = 0.37572813, f1_score = 0.81646438, precision = 0.78565255, recall = 0.88257487, grad_params:norm() = 1.2556e+00[0m	
[0mdev set:   loss = 5.64212655, f1_score = 0.25770454, precision = 0.26013115, recall = 0.25809553[0m	
[0mtrain set: loss = 0.49148154, f1_score = 0.65898251, precision = 0.62987881, recall = 0.82804315, grad_params:norm() = 1.6640e+00[0m	
[0mdev set:   loss = 5.67854709, f1_score = 0.25932898, precision = 0.25882361, recall = 0.26240873[0m	
[0mtrain set: loss = 0.40021737, f1_score = 0.85581286, precision = 0.85515608, recall = 0.85821851, grad_params:norm() = 4.4860e-01[0m	
[0mdev set:   loss = 5.75015865, f1_score = 0.25900874, precision = 0.25997797, recall = 0.26184089[0m	
[0mtrain set: loss = 0.36129054, f1_score = 0.82461562, precision = 0.79386501, recall = 0.89055298, grad_params:norm() = 1.2625e+00[0m	
[0mdev set:   loss = 5.80758584, f1_score = 0.26171921, precision = 0.26514217, recall = 0.26170220[0m	
```

