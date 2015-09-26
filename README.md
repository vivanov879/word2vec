Torch implementation of word2vec and sentiment analysis as explained in assignment1.pdf. Negative sampling loss function and max-margin learning are used for word2vec. 

- In Terminal.app, run ```python extract_datasets_for_torch.py``` to generate datasets for sentiment_analysis.lua.
- Run ```th filter_sentences.lua``` to generate vocabulary of words and then write sentences to file using this vocabulary
- Run ```th word2vec.lua``` to train the word2vec model.
- Run ```th visualize_word_vectors.lua``` to project several words into 2d space using SVD and then plot this words. You'll get something like Result_example.png.
- Run ```th gen_sentiment_files.lua``` generates training and dev sets of words and their labels
- Run ```th sentiment_analysis.lua``` trains a sentiment analysis model.

You can use the program with your own text. ``` filter_sentences.lua``` works with tokenized input text. You can tokenize you text using https://github.com/moses-smt/mosesdecoder by running 
```
cd ~/mosesdecoder/scripts/tokenizer/ 
./pre-tokenizer.perl ~/word2vec/trainSentences_raw | ./lowercase.perl | ./tokenizer.perl > ~/word2vec/trainSentences
``` 
