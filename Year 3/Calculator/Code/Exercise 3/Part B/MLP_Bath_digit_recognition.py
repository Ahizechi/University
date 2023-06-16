# -*- coding: utf-8 -*-
"""
@author: Ahizechi Nwankwo

Description: Training MLP for recognition of digits: 0 to 9
             The trained model will be saved and used by another program
             to recognise digits in real-time and control a process via ROS
             
             This program uses a training and testing data collected
             at the University of Bath.
             
             Do NOT use the MNIST dataset in this program
"""

import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import os

from PIL import Image                                                            
import glob

# seed for repeatibility
np.random.seed(7)

# define the list of digits in the training/testing dataset
namesList = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# path to folders with training and testing images
# imageFolderPath = r'.'
imageFolderTrainingPath = r'/u/i/an777/Bath_digits_dataset/train'   # adjust to your folder with training data
imageFolderTestingPath = r'/u/i/an777/Bath_digits_dataset/validation' # adjust to your folder with testing data
imageTrainingPath = []
imageTestingPath = []

# search for all images .jpg in the folders defined above
for i in range(len(namesList)):
    trainingLoad = imageFolderTrainingPath + '/' + namesList[i] + '/*.jpg'
    testingLoad = imageFolderTestingPath + '/' + namesList[i] + '/*.jpg'
    imageTrainingPath = imageTrainingPath + glob.glob(trainingLoad)
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)
    
# print number of training and testing images
print(len(imageTrainingPath))
print(len(imageTestingPath))

# new width and height for the images
imWidth = 28
imHeight = 28
multipliedWH = 784

# create array zeros for training and testing data
x_train = np.zeros((len(imageTrainingPath), imHeight, imWidth))
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth))

# load image matrix in each position of x_train and x_test arrays
print('Loading Train Images...')
for i in range(len(x_train)):
    tempImg = Image.open(imageTrainingPath[i]).convert('L')
    tempImg = tempImg.resize((imWidth, imHeight))
    x_train[i, :, :] = np.array(tempImg, 'f')
  
print('Loading Test Images...')
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg = tempImg.resize((imWidth, imHeight))
    x_test[i, :, :] = np.array(tempImg, 'f')

# create vector of zeros for training and testing targets (or labels)
print('Creating Vectors...')
y_train = np.zeros((len(x_train),));
y_test = np.zeros((len(x_test),));

# add training labels to the corresponding numbers
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTrainingPath)/len(namesList))):
        y_train[countPos,] = i
        countPos = countPos + 1
    
# add testing labels to the corresponding numbers
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
###############################################################
# obtain here the training and testing labels in onehot format
###############################################################

# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("Test labels: ", dict(zip(unique, counts)))

# compute the number of labels
num_labels = len(np.unique(y_train))
print(num_labels)

# convert to one-hot vector
y_train_onehot = tf.keras.utils.to_categorical(y_train)
print(y_train_onehot)
y_test_onehot = tf.keras.utils.to_categorical(y_test)
print(y_test_onehot)
  
# image dimensions
image_size = x_train.shape[1]
input_size = image_size * image_size

# resize and normalize
x_train_flattened = np.reshape(x_train, [-1, input_size])
x_test_flattened = np.reshape(x_test, [-1, input_size])

# mean centering and normalization:
mean_vals = np.mean(x_train_flattened, axis=0)
std_val = np.std(x_train_flattened)
	
# use mean and std to center data and normalise
x_train_centered = (x_train_flattened - mean_vals)/std_val
x_test_centered = (x_test_flattened - mean_vals)/std_val
print(x_train_centered)
print(x_test_centered)


print('First 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])


# initialize model
model = tf.keras.models.Sequential()
	
# add input layer
model.add(tf.keras.layers.Dense(units=50, input_dim=x_train_centered.shape[1], activation='sigmoid'))

# add hidden layer
model.add(tf.keras.layers.Dense(units=50, input_dim=50, activation='sigmoid'))
	
# add output layer
model.add(tf.keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=50, activation='softmax'))
	
# define SGD optimizer
sgd_optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)

# compile model
model.compile(optimizer=sgd_optimizer,loss='categorical_crossentropy')

model.summary()

# train model
history = model.fit(x_train_centered, y_train_onehot, epochs=50)

# calculate training accuracy
y_train_pred = model.predict_classes(x_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
	
print(f'Training accuracy: {(train_acc * 100):.2f}')
	

# calculate testing accuracy
y_test_pred = model.predict_classes(x_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
	
print(f'Test accuracy: {(test_acc * 100):.2f}')

###############################################################
# flatten the training and testing input data 
###############################################################

# add your code here

# imWidth = 28
# imHeight = 28


###############################################################
# center and normalise the training and testing input data
###############################################################

# add your code here


###############################################################
# create your MLP network using a sequential model with Keras
# First try with 1 input layer, 1 hidden layer and 1 output layer and check the accuracy
# If accuracy is low then try changing the number of epoch, batch_size and adding hidden layers
# Show here the summary of your MLP network
###############################################################

# add your code here for the following:
# Create your MLP network
# Use the SGD optimizer function to train your model
# Summary of your network
# Check accuracy for training and testing
# Save your model, which will be use for recognition and control

