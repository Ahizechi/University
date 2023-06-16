import numpy as np

def perceptron(inputs_list, weights_list, bias):
    
    inputs = np.array(inputs_list).all
    weights = np.array(weights_list).all
    summed = np.dot(inputs, weights)
    summed = summed + bias

    if summed >= 0:
        output = 1
    else:
        output = 0
    return output

inputs = ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])
weights = [1.0, 1.0]
bias = -1.5

for x in inputs:
   print("Result: ", perceptron(inputs, weights, bias)) 

print("Inputs: ", inputs)
print("Weights: ",weights)
print("Bias: ", bias)
print("Result: ", perceptron(inputs, weights, bias))