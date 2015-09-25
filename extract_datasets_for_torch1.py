
import random
import numpy as np
from cs224d.data_utils import *


random.seed(1)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

num_to_word = {}
word_to_num = tokens

for (k, v) in word_to_num:
    num_to_word[v] = k




def extract(trainset, fn_words, fn_labels):

    nTrain = len(trainset)
    trainLabels = [None for _ in range(nTrain)]
    trainSentences = [None for _ in range(nTrain)]

    for i in xrange(nTrain):
        trainWords, trainLabel = trainset[i]
        trainLabels[i] = str(trainLabel + 1) + '\n'
        trainSentences[i] = ' '.join(trainWords) + '\n'

    with open(fn_words, 'w') as f:
        f.writelines(trainSentences)
    with open(fn_labels, 'w') as f:
        f.writelines(trainLabels)

b = 1
pass

extract(dataset.getTrainSentences(), 'trainSentences_raw', 'trainLabels')
extract(dataset.getDevSentences(), 'devSentences_raw', 'devLabels')

