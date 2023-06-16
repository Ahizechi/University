# Method of detecting spikes in the d vector, generating the index vector, and classifying each spike. The code was 
# improved compared to task 1, and the spike detection is much better. This task makes use of an artifical neural
# network to classify the different spikes. 
# 
# An artificial neural network (ANN) is a type of machine learning algorithm modeled after the structure and function of the human brain. 
# It is composed of a large number of interconnected "neurons," which can process and transmit information. ANNs are designed to recognize 
# patterns and make predictions or decisions based on input data.
# 
# The basic building blocks of an ANN are the input layer, hidden layers, and output layer. The input layer receives the input data and passes 
# it through the hidden layers, which are responsible for learning and extracting features from the data. 
# The output layer produces the final prediction or decision based on the information processed in the hidden layers.
#
# The Median absolute deviation was optimised using the Iglewicz and Hoaglin formula, where the MAD will converge to the 
# median of the half normal distribution, which is the 75% percentile of a normal distribution, and N(0.75)≃0.6745
# The training data is collected from D1, tested to ensure its working correctly, then tested using the spikes obtained in D2. 
# The final output is a .mat file called D2 that contains both the index and the class of each spike.

# Import the scipy.io module as spio
import scipy.io as spio

# Import the matplotlib.pyplot module as plt
import matplotlib.pyplot as plt

# Import the numpy module as np
import numpy as np

# Import the butter, sosfiltfilt, and find_peaks functions from the scipy.signal module
from scipy.signal import butter, sosfiltfilt, find_peaks

# Import the train_test_split function from the sklearn.model_selection module
from sklearn.model_selection import train_test_split

# Import the metrics module from sklearn
from sklearn import metrics

# Import the Sequential, Dense and Input classes from the keras.models module
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

# Import the Adam optimizer from the keras.optimizers module
from keras.optimizers import Adam


def filt_signal(signal, cutoff, type, fs=25000, order=2):
    # Define the filter coefficients using the butter function
    filtered_values = butter(order, cutoff, btype=type, analog=False, output='sos', fs=fs)

    # Apply the filter to the input signal using the sosfiltfilt function
    filtered_signal = sosfiltfilt(filtered_values, signal)

    # Return the filtered signal
    return filtered_signal


def peak_find(signal, prominence):
    # Compute the median absolute deviation of the signal
    mad = np.median(np.absolute(signal)/0.6745) 

    # Set the threshold for peak detection based on the median absolute deviation
    threshold = 5 * mad

    # Use the find_peaks function to detect peaks in the signal
    peaks, _ = find_peaks(signal, height=threshold, prominence=prominence)

    # Return the detected peaks
    return peaks


def spike_removal(signal, peaks, window_size, type, find_index_val=0, neuron_val=0):

    spikes = []
    
    # If the type is 'training', then we need to extract the spikes and their corresponding classes
    if type == 'training_data':

        training_class = [] 
        repeat_peaks = [] 
        
        # For each peak index, extract the corresponding spike and its class
        for peak_find_index_val in peaks:
            
            # Get the index of the peak that is closest to but not greater than the current peak index
            i = find_index_val[find_index_val < peak_find_index_val].max()
            
            # Skip this peak if it is a duplicate
            if i in repeat_peaks: 
                continue
            repeat_peaks.append(i)
            
            # Get the index of the neuron that corresponds to this peak
            n = np.where(find_index_val==i)[0][0]
            
            # Extract the spike from the signal using the specified window size
            extract_val = signal[peak_find_index_val-window_size[0]:peak_find_index_val+window_size[1]]
            spikes.append(extract_val)
            training_class.append(neuron_val[n])

        return spikes, training_class

    # If the type is 'testing_data', then we only need to extract the spikes
    if type == 'testing_data':
        
        # For each peak index, extract the corresponding spike
        for peak_find_index_val in peaks:
            
            # Extract the spike from the signal using the specified window size
            extract_val = signal[peak_find_index_val-window_size[0]:peak_find_index_val+window_size[1]]
            spikes.append(extract_val)
            
        return spikes

    # If the type is not 'training' or 'testing_data', then return an empty list
    return spikes


def identify(training_path, test_path, window_size, training_frequency):

    # Unpack the dictionary containing the training data
    training = training_path

    # Unpack the dictionary containing the test data
    testing = test_path

    # Get the indices and classes of the training data
    Index = training['Index']
    Class = training['Class']

    # Get the training and test signals
    training_signal = training['d']
    testing_signal = testing['d']

    # Filter the training signal using a low-pass filter with the specified cutoff frequency
    training_filter = filt_signal(training_signal, training_frequency, 'low')

    # Detect peaks in the filtered training signal
    training_peaks = peak_find(training_filter, prominence=0.2)

    # Extract the spikes from the filtered training signal and their corresponding classes
    training_spikes, training_classes = spike_removal(training_filter, training_peaks, window_size, 'training_data', Index, Class)

    # Filter the test signal using a low-pass filter with the specified cutoff frequency
    testing_filter = filt_signal(testing_signal, training_frequency, 'low')

    # Detect peaks in the filtered test signal
    testing_peaks = peak_find(testing_filter, prominence=0.2)

    # extract the spikes from the filtered test signal
    testing_spikes = spike_removal(testing_filter, testing_peaks, window_size, 'testing_data')

    return training_spikes, training_classes, testing_peaks, testing_spikes


def one_hot_rep(conv_lab):
    
    # Convert the integer labels from a range starting from 1 to the number of classes
    conv_lab = np.subtract(conv_lab, 1)

    # Get the number of classes
    Number_Class = np.max(conv_lab) + 1

    # Convert the labels to a one-hot encoding
    one_hot = np.eye(Number_Class)[conv_lab.reshape(-1)].T
    one_hot = one_hot.T

    return one_hot


def neural_network_ANN(input_layer, output_layer, hidden_layer, learning_rate):
    
    # Define the architecture of the neural network
    model = Sequential()

    # Add an input layer to the model
    model.add(Input(shape=input_layer))

    # Add a hidden layer with hidden_layer units and ReLU activation function
    model.add(Dense(hidden_layer, activation='relu'))

    # Add an output layer with output_layer units and softmax activation function
    model.add(Dense(output_layer, activation='softmax'))

    # Compile the model with categorical crossentropy loss function, Adam optimizer, and specified learning rate
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return model


# Define the training and test data paths
training_path = spio.loadmat('D1(1).mat', squeeze_me=True)
test_path = spio.loadmat('D2(1).mat', squeeze_me=True) 
initial_test_signal = test_path['d']


# Define the window_size size and cut off frequency for spike detection
window_size = [15, 24]
training_frequency = 2500


# Obtain the training and test data
training_spikes, training_classes, testing_peaks, testing_spikes = identify(training_path, test_path, window_size, training_frequency)
X, Y, N = training_spikes, training_classes, testing_spikes
Test_IndexVal = testing_peaks


# Convert to arrays
X, Y, N = np.asarray(X), np.asarray(Y), np.asarray(N)


# Change classes to one hot representation
Y = one_hot_rep(Y)


# Split training dataset - This was split 80-20.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# Define the parameters for the neural network - optimal values were found
input_layer = X_train.shape[1]
output_layer = Y_train.shape[1]
hidden_layer = 30
learning_rate = 0.01
epochs = 120
batch_size = 32


# Create and train the neural network model
model = neural_network_ANN(input_layer, output_layer, hidden_layer, learning_rate)
train_model = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)


# Predict class spikes in training data. This was done to test the accuracy of the algorithm
train_test_class = model.predict(X_test, batch_size=batch_size)
train_test_class = (train_test_class == train_test_class.max(axis=1)[:,None]).astype(int)


# Predict class spikes in test data
TestPred_Class = model.predict(N, batch_size=batch_size)
TestPred_Class = (TestPred_Class == TestPred_Class.max(axis=1)[:,None]).astype(int)


# Convert one hot into classes and store these values in a .mat file
Neuron_Class = [np.where(v == 1)[0][0] + 1 for v in TestPred_Class]
spio.savemat('D2.mat', mdict={'d': initial_test_signal, 'Index': Test_IndexVal, 'Class': Neuron_Class})