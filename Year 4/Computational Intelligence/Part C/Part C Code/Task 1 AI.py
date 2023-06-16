# Very simple method of detecting spikes in the d vector and generating the index vector. The code would need to 
# be improved for further tasks as it currently does not detect all peaks. This can be achieved by changing the
# filter type, and changing the way the threshold and median absolute deviation is calculated. The peaks that are
# detect are within the +/- 50 range of the respective spikes when testing with dataset D1.

# Import the scipy.io module as spio
import scipy.io as spio

# Import the matplotlib.pyplot module as plt
import matplotlib.pyplot as plt

# Import the numpy module as np
import numpy as np

# load data from .mat file
data = spio.loadmat('D2(1).mat', squeeze_me=True)

# get signal from data
signal = data['d']

# calculate median absolute deviation and determine the threshold value
mad = np.median(np.abs(signal - np.median(signal)))
threshold = 7 * mad

# filter the signal using an averaging filter
filtered_signal = np.convolve(signal, np.ones(10)/10, mode='same')

# find indices of spikes
spike_indices = np.where(np.abs(filtered_signal) >= threshold)[0]

# find the start of each spike by finding the first sample
# above the threshold in each spike
IndexVal = [i[0] for i in np.split(spike_indices, np.where(np.diff(spike_indices) != 1)[0]+1) if len(i) > 0]

# save to D2.mat file, the name is changed to avoid confusion with task 2
spio.savemat('D2(TaskA).mat', mdict={'d': signal, 'Index': IndexVal})
