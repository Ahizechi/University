# -- coding: utf-8 --
"""
@author: Uriel Martinez-Hernandez

EE30241 - Robotics and Autonomous Systems

The perceptron model

Perceptron with 3 inputs for detection of red and green colours
and control of the Pioneer robot in CoppeliaSim
"""

import numpy as np
from random import seed
from random import random
from random import randint
import sys
from sys import exit
import time

import sim as vrep # access all the VREP elements

vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # start aconnection
if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")


returnCode,leftMotorHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_blocking)
returnCode,rightMotorHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_blocking)
returnCode,robotCameraHandle = vrep.simxGetObjectHandle(clientID,'Vision_sensor1',vrep.simx_opmode_blocking)


# define random values for Weights
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

threshold = 5 # set threshold 
epoch = 500 # number of epochs 

for i in range (0, epoch):
    for j in range (len(Y)):
           
        sum = X[j][0]*Weights[0] + X[j][1]*Weights[1] +  X[j][2]*Weights[2]  + Bias
        
        actual=Y[j]
        
        if sum > threshold:
            predicted = 1
        else:
            predicted = 0
        
        delta = actual - predicted
        
        Weights[0] = Weights[0] + delta * lRate * X[j][0] # update weights
        Weights[1] = Weights[1] + delta * lRate * X[j][1]
        Weights[2] = Weights[2] + delta * lRate * X[j][2]
        
        Bias = Bias + lRate + delta # update bias

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


# Get data from Coppelia for recognition and control - first call
returnCode,resolution,rgbImage = vrep.simxGetVisionSensorImage(clientID,robotCameraHandle,0,vrep.simx_opmode_streaming)
[returnCodeVision,rgbRobotImageData,rgbRobotVectorData] =vrep.simxReadVisionSensor(clientID,robotCameraHandle,vrep.simx_opmode_streaming)

err_code = vrep.simxSetJointTargetVelocity(clientID,leftMotorHandle,0.3,vrep.simx_opmode_streaming);
err_code = vrep.simxSetJointTargetVelocity(clientID,rightMotorHandle,-0.3,vrep.simx_opmode_streaming);


# iterate to control the robot in CoppeliaSim
for i in range(10000):
    # Get data from Coppelia for recognition and control
    returnCode,resolution,rgbRobotImage = vrep.simxGetVisionSensorImage(clientID,robotCameraHandle,0,vrep.simx_opmode_buffer)
    [returnCodeVision,rgbRobotImageMatrix,rgbRobotVectorData] = vrep.simxReadVisionSensor(clientID,robotCameraHandle,vrep.simx_opmode_buffer)

    # check if there is data ready from the camera
    if( rgbRobotVectorData ):

        # split the list of values in rgbRobotVectorData and convert to float values
        tmpImgData = np.zeros((15,1))
        for idx, item in enumerate(rgbRobotVectorData[0]):
            tmpImgData[idx,0] = item
    
        # now the vector has float values
        rgbRobotVectorData = tmpImgData;
                
        time.sleep(0.1)

        if( rgbRobotVectorData[0,0] > 0 ):
            coppeliaSimOutput = rgbRobotVectorData[1:4,0].dot(Weights[:,-1:]) + Bias[-1]

            # if class = 0 (or object is red colour) then increase velocity
            if( coppeliaSimOutput < 0 ):
                coppeliaSimOutput = 0
                print('RED BLOCK')
                err_code = vrep.simxSetJointTargetVelocity(clientID,leftMotorHandle,1.5,vrep.simx_opmode_streaming)
                err_code = vrep.simxSetJointTargetVelocity(clientID,rightMotorHandle,-1.5,vrep.simx_opmode_streaming)

            # if class = 1 (or object is green colour) then reduce velocity
            else:
                coppeliaSimOutput = 1
                print('GREEN BLOCK')
                err_code = vrep.simxSetJointTargetVelocity(clientID,leftMotorHandle,0.2,vrep.simx_opmode_streaming)
                err_code = vrep.simxSetJointTargetVelocity(clientID,rightMotorHandle,-0.2,vrep.simx_opmode_streaming)

        # when the camera doesn't detect an object
        # move the robot a specific velocity
        else:
            err_code = vrep.simxSetJointTargetVelocity(clientID,leftMotorHandle,0.4,vrep.simx_opmode_streaming)
            err_code = vrep.simxSetJointTargetVelocity(clientID,rightMotorHandle,-0.4,vrep.simx_opmode_streaming)
            print('NO OBJECT DETECTED')
    else:
        print('NO OBJECT DETECTED')

vrep.simxFinish(clientID);