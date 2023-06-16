# Method of detecting spikes in the d vector, generating the index vector, and classifying each spike. The code was 
# improved compared to task 1, and the spike detection is much better. This task makes use of an k-nearest neighbors algorithm
# to classify the different spikes. 
# 
# K-nearest neighbors (KNN) is a type of machine learning algorithm that can be used for classification and regression tasks. 
# It works by identifying the K number of data points in a dataset that are closest (in terms of distance) to a given point, 
# and then using the class labels or values of those points to make a prediction about the class label or value of the given point.
# 
# To classify a new data point using KNN, the algorithm calculates the distance between the new point and each of the points in the dataset. 
# It then selects the K points that are closest to the new point and determines the most common class label among those K points. 
# The new point is then assigned the class label that is most common among its K nearest neighbors.
# 
# The Median absolute deviation was optimised using the Iglewicz and Hoaglin formula.
# The MAD will converge to the median of the half normal distribution, which is the 75% percentile of a normal distribution, and N(0.75)≃0.6745
# The training data is collected from D1, tested to ensure its working correctly, then tested using the spikes obtained in D3. 
# The final output is a .mat file called D3 that contains both the index and the class of each spike.

# Import the scipy.io module as spio
import scipy.io as spio

# Import the matplotlib.pyplot module as plt
import matplotlib.pyplot as plt

# Import the numpy module as np
import numpy as np

# Import the scipy module
import scipy.signal

# Import the butter, sosfiltfilt, and find_peaks functions from the scipy.signal module
from scipy.signal import butter, sosfiltfilt, find_peaks

# Import the train_test_split function from the sklearn.model_selection module
from sklearn.model_selection import train_test_split

# Import the metrics module from sklearn
from sklearn import metrics

# Import scikit-learn k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier


def filt_signal(signal, cutoff, type, fs=25000, order=2):
    # Define the filter coefficients using the butter function
    filtered_values = butter(order, cutoff, btype=type, analog=False, output='sos', fs=fs)

    # Apply the filter to the input signal using the sosfiltfilt function
    filtered_signal = sosfiltfilt(filtered_values, signal)

    # Return the filtered signal
    return filtered_signal


def peak_find(signal, prominence=0):
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


def identify(training_path, test_path, window_size, training_frequency, norm_frequency, test_frequency):

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
    training_peaks = peak_find(training_filter)

    # Extract the spikes from the filtered training signal and their corresponding classes
    training_spikes, training_classes = spike_removal(training_filter, training_peaks, window_size, 'training_data', Index, Class)

    # Filter the test signal using a low-pass filter with the specified cutoff frequency
    norm_signal = filt_signal(testing_signal, norm_frequency, 'high')

    # Denoise the training recording with a low-pass filter
    testing_filter = filt_signal(norm_signal, test_frequency, 'low')    

    # Detect peaks in the filtered test signal
    testing_peaks = peak_find(testing_filter, prominence=0.2)

    # extract the spikes from the filtered test signal
    testing_spikes = spike_removal(testing_filter, testing_peaks, window_size, 'testing_data')

    return training_spikes, training_classes, testing_peaks, testing_spikes


# Define the training and test data paths
training_path = spio.loadmat('D1(1).mat', squeeze_me=True)
test_path= spio.loadmat('D3(1).mat', squeeze_me=True)
initial_test_signal = test_path['d']


# Display Power Spectrum to calculate cut off frequency for the test signal
# Load the signal from the .mat file
# Make sure to set squeeze_me=True when loading the .mat file
power_spectrum = spio.loadmat('D3(1).mat', squeeze_me=True)['d']

# Set the sampling frequency and the cut-off frequency
freqs = 25e3
cutoff = 25

# Create the Butterworth high pass filter
bb, aa = scipy.signal.butter(3, cutoff, 'highpass', fs=freqs)

# Apply the filter to the signal using lfilter
power_spec_filt = scipy.signal.lfilter(bb, aa, power_spectrum)

# Calculate and plot the power spectrum of the filtered signal
f, Pxx = scipy.signal.periodogram(power_spec_filt, fs=freqs)
# plt.plot(f, Pxx). This will display the power spectrum graph. Has been commented out because this value has been obtained.
# plt.show()


# Define the window_size size and frequencies for spike detection
window_size = [15, 24]
training_frequency = 2500
norm_frequency = 25
test_frequency = 2000


# Obtain the training and test data
training_spikes, training_classes, testing_peaks, testing_spikes = identify(training_path, test_path, window_size, training_frequency, norm_frequency, test_frequency)
X, Y, N = training_spikes, training_classes, testing_spikes
Test_IndexVal = testing_peaks


# Split training dataset - This was split 80-20.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# Define KNN parameters
k = 3
p = 2 


# Create the KNN model
model = KNeighborsClassifier(n_neighbors = k, p = p)
model.fit(X_train, Y_train)


# Predict classes for spikes in the training subset
y_predict = model.predict(X_test)


# Predict classes for spikes in the test subset
Neuron_Class = model.predict(N)


# Export testing data as .mat file
spio.savemat('D3.mat', mdict={'d': initial_test_signal, 'Index': Test_IndexVal, 'Class': Neuron_Class})
