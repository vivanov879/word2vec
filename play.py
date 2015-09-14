
import random
import numpy as np
from cs224d.data_utils import *


dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext
print(dataset.getRandomContext(2))



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
    print(dataset.getRandomTrainSentence())
    print(dataset.getTrainSentences()[random.randint(1, 1000)])

b = 1
pass



