
import random
import numpy as np
from cs224d.data_utils import *


random.seed(1)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 10

for _ in range(10):
    print(dataset.getRandomContext(C))
    #print(dataset.getRandomTrainSentence())
    #print(dataset.getTrainSentences()[random.randint(1, 1000)])

b = 1
pass
