import numpy as np

def bin_array(array, bins=100):
    len_data = len(array)
    x = np.arange(len_data)+1
    num_bin = len_data//bins
    data_binned = []
    x_binned = []
    for i in range(num_bin):
        data_binned.append(np.mean(array[i*bins:(i+1)*bins]))
        x_binned.append(np.mean(x[i*bins:(i+1)*bins]))
    data_binned = np.array(data_binned)
    x_binned = np.array(x_binned)
    return x_binned, data_binned
