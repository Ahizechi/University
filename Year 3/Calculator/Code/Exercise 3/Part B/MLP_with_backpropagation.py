# -*- coding: utf-8 -*-
"""
@author: Ahizechi Nwankwo

EE30241 - Robotics and Autonomous Systems

Multilayer Perceptron (MLP) network with backpropagation algorithm

MLP network to recognise XOR input pattern
"""

import numpy as np
from random import seed
from random import random
from random import randint
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

x=np.array([[0,0,1,1],[0,1,0,1]]) # XOR Inputs

y=np.array([[0,1,1,0]]) # XOR Outputs

input_x = 2 # Number of Inputs

input_y = 1 # Number of Neurons in Output

neurons_z = 2 # Number of Neurons in Hidden Layer

m = x.shape[1] # Training

lr = 0.5 # Learning rate

np.random.seed(2)

w1 = np.random.rand(neurons_z,input_x)   # Weight matrix for hidden layer
w2 = np.random.rand(input_y,neurons_z)   # Weight matrix for output layer

b3 = np.random.uniform(-0.5, 0.5, size=(1))
b4 = np.random.uniform(-0.5, 0.5, size=(1))
b5 = np.random.uniform(-0.5, 0.5, size=(1))

losses = [] # Losses

def sigmoid(z): # Use sigmoid for hidden layer and output
    z= 1/(1+np.exp(-z))
    return z

def forward_prop(w1,w2,x): # Forward propagation
    z1 = np.dot(w1,x)
    a1 = sigmoid(z1)    
    z2 = np.dot(w2,a1)
    a2 = sigmoid(z2)
    return z1,a1,z2,a2

def back_prop(m,w1,w2,z1,a1,z2,a2,y): # Backward propagation
    
    dz2 = a2-y
    dw2 = np.dot(dz2,a1.T)/m
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1)
    dw1 = np.dot(dz1,x.T)/m
    dw1 = np.reshape(dw1,w1.shape)
    
    dw2 = np.reshape(dw2,w2.shape)    
    return dz2,dw2,dz1,dw1

iterations = 5000
for i in range(iterations):
    z1,a1,z2,a2 = forward_prop(w1,w2,x)
    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    w2 = w2-lr*dw2 # Correct Weights with Bias
    w1 = w1-lr*dw1

plt.figure() # Plot losses
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Learning Error")
plt.title("Learning Error vs Epochs")
plt.show()

def predict(w1,w2,input):
    z1,a1,z2,a2 = forward_prop(w1,w2,test)
    value = np.squeeze(a2)
    if value>=0.5:
        print("For input", [i[0] for i in input], "output is 1")
    else:
        print("For input", [i[0] for i in input], "output is 0")

test = np.array([[0],[0]])
predict(w1,w2,test)
test = np.array([[0],[1]])
predict(w1,w2,test)
test = np.array([[1],[0]])
predict(w1,w2,test)
test = np.array([[1],[1]])
predict(w1,w2,test)
