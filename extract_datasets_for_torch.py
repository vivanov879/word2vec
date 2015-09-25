
import random
import numpy as np
from cs224d.data_utils import *


random.seed(1)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

num_to_word = {}
word_to_num = tokens

for (word, index) in tokens.items():
    num_to_word[index] = word
    word_to_num[word] = index


def extract(trainset, fn_words, fn_labels):

    nTrain = len(trainset)
    trainLabels = [None for _ in range(nTrain)]
    trainSentences = [None for _ in range(nTrain)]

    X_train = []
    y_train = []

    for i in xrange(nTrain):
        trainWords, trainLabel = trainset[i]
        trainWords = [word_to_num[word] for word in trainWords]
        y_train.append(str(trainLabel + 1) + '\n')
        X_train.append(' '.join([str(k) for k in trainWords]) + '\n')

    with open(fn_words, 'w') as f:
        f.writelines(X_train)
    with open(fn_labels, 'w') as f:
        f.writelines(y_train)


b = 1
pass


with open('inv_vocabulary_raw', 'w') as f1:
    with open('vocabulary_raw', 'w') as f2:
        lines1 = []
        lines2 = []
        for word, num in word_to_num.items():
            lines1.append(word + ' ' + str(num+1) + '\n')
            lines2.append(str(num+1) + ' ' + word + '\n')
        f1.writelines(lines1)
        f2.writelines(lines2)

extract(dataset.getTrainSentences(), 'x_train', 'y_train')
extract(dataset.getDevSentences(), 'x_dev', 'y_dev')
