# -*- coding: utf-8 -*-
"""
@author: Uriel Martinez-Hernandez

EE30241 - Robotics and Autonomous Systems

The perceptron model

Perceptron with 3 inputs for detection of red and green colours
"""

import numpy as np
from random import seed
from random import random
from random import randint

## define random values for Weights
Weights = np.random.uniform(-0.5, 0.5, size=(3,1))
# define random values for Bias
Bias = np.random.uniform(-0.5, 0.5, size=(1))
# define value for learning rate
lRate = 0.1

print('Initial weights and bias')
print('weights: ', Weights)
print('bias: ', Bias)
print('learning rate: ', lRate)

# define input (X) and targets (Y)
X = np.random.uniform(0.0, 0.5, size=(1000,3))
X[0:500,:-2] = np.random.uniform(0.8, 1.0, size=(500,1))
X[500:1000,1:-1] = np.random.uniform(0.8, 1.0, size=(500,1))

Y = np.zeros((len(X),1))
Y[int(len(Y)/2):] = 1

threshold = 5
epoch = 500

for i in range (0, epoch):
    for j in range (len(Y)):
           
        sum = X[j][0]*Weights[0] + X[j][1]*Weights[1] +  X[j][2]*Weights[2]  + Bias
        
        actual=Y[j]
        
        if sum > threshold:
            predicted = 1
        else:
            predicted = 0
        
        delta = actual - predicted
        
        Weights[0] = Weights[0] + delta * lRate * X[j][0]
        Weights[1] = Weights[1] + delta * lRate * X[j][1]
        Weights[2] = Weights[2] + delta * lRate * X[j][2]
        
        Bias = Bias + lRate + delta

Xtest = np.random.uniform(0.0, 0.5, size=(1000,3))
Xtest[0:500,:-2] = np.random.uniform(0.8, 1.0, size=(500,1))
Xtest[500:1000,1:-1] = np.random.uniform(0.8, 1.0, size=(500,1))

Ytest = Y
for j in range (len(Y)):
           
    sum = Xtest[j][0]*Weights[0] + Xtest[j][1]*Weights[1] +  Xtest[j][2]*Weights[2] +  Bias
        
    if sum > threshold:
        Ytest[j] = 1
    else:
        Ytest[j] = 0

