# -*- coding: utf-8 -*-
"""
EE30241 - Robotics and Autonomous Systems

The perceptron model

Perceptron with 2 inputs for detection of AND, OR logic gates
"""
# perceptron 2-inputs

import numpy as np
from random import seed
from random import random
from random import randint
import matplotlib.pyplot as plt


weights = np.random.uniform(-0.5, 0.5, size=(2,1))
bias = np.random.uniform(-0.5, 0.5, size=(1))
lRate = 0.1


print('Initial weights and bias')
print('weights: ', weights)
print('bias: ', bias)
print('learning rate: ', lRate)

operator = input('Opertator: ')

X = np.array([[0,0],[0,1],[1,0],[1,1]]) # define inputs

if operator == 'and':
	Y = np.array([0, 0, 0, 1]) # define and gate output
elif operator == 'or':
	Y = np.array([0, 1, 1, 1]) # define or gate output
elif operator == 'nor':
	Y = np.array([1, 0, 0, 0]) # define nor gate output

w = weights
threshold = 5
epoch = 1000 #learning time

costFunc = np.ones(1)
print(costFunc)

for i in range(0, epoch):
	print("epoch ", i+1)
	global_delta = 0 # terminate loop if completed early
	for j in range(len(X)):
		
		actual = Y[j]
		
		sum = X[j][0]*w[0] + X[j][1]*w[1] + bias
		
		if sum > threshold: # then fire
			predicted = 1
		else: # do not fire
			predicted = 0
		
		delta = actual - predicted # error
		global_delta = global_delta + abs(delta)
		
		#update weights and bias with respect to the error
		w[0] = w[0] + delta * lRate * X[j][0]
		w[1] = w[1] + delta * lRate * X[j][1]
		bias = bias + lRate * delta
				
		print(operator,"[", X[j][0], X[j][1],"]", "-> Expected:", actual, ", Prediction:", predicted, " (w:",w[0],")", " (b:",bias,")")
		
	if global_delta == 0: # stop the code once the code has learnt
		break
	
	print("------------------------------")

output = np.zeros((len(Y),1))

Xtest = X
output_test = np.zeros((len(Y),1))




