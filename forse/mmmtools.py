import numpy as np

def split_training_set(xraw):
    nstamps = xraw.shape[-1]
    npix = xraw.shape[0]
    nchans = 1
    ntrains = int(nstamps *  4./5.)
    ntests = int(nstamps * 1./5.)
    train = xraw[:,:,:ntrains].T.reshape(ntrains,npix,npix, 1)
    test = xraw[:,:,-ntests:].T.reshape(ntests,npix,npix,  1)
    return train, test

def divide_image(image):
    test_set_zoom = np.zeros((25, 64, 64))
    ind = 0
    for xax in range(5):
        for yax in range(5):
            test_set_zoom[ind, :, :] = image[64*xax:64*(xax+1), 64*yax:64*(yax+1)]
            ind = ind+1
    return test_set_zoom

def unify_image(images):
    one_big = np.zeros((320, 320))
    ind = 0
    for xax in range(5):
        for yax in range(5):
            one_big[64*xax:64*(xax+1), 64*yax:64*(yax+1)] = images[ind, :, :, 0]
            ind = ind+1
    return one_big

def bin_history(history, bins=100):
    len_data = len(history)
    x = np.arange(len_data)+1
    num_bin = len_data//bins
    data_binned = []
    x_binned = []
    for i in range(num_bin):
        data_binned.append(np.mean(history[i*bins:(i+1)*bins]))
        x_binned.append(np.mean(x[i*bins:(i+1)*bins]))
    data_binned = np.array(data_binned)
    x_binned = np.array(x_binned)
    return x_binned, data_binned
