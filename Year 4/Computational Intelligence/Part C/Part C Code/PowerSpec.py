import scipy.signal
import scipy.io as spio
import matplotlib.pyplot as plt

# Load the signal from the .mat file
# Make sure to set squeeze_me=True when loading the .mat file
signal = spio.loadmat('D2(1).mat', squeeze_me=True)['d']

# Set the sampling frequency and the cut-off frequency
fs = 25e3
cutoff = 25

# Create the Butterworth high pass filter
b, a = scipy.signal.butter(3, cutoff, 'highpass', fs=fs)

# Apply the filter to the signal using lfilter
filtered_signal = scipy.signal.lfilter(b, a, signal)

# Calculate and plot the power spectrum of the filtered signal
f, Pxx = scipy.signal.periodogram(filtered_signal, fs=fs)
plt.plot(f, Pxx)
plt.show()
