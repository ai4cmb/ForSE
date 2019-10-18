import numpy as np
import healpy as hp

def load_training_set(patches_file, part_train=0.8, part_test=0.2, part_val=None, seed=4324, reshape=True):
    Y,X = np.load(patches_file)
    Y = Y/X
    for i in range(Y.shape[0]):
        Y[i] = rescale_min_max(Y[i])
        X[i] = rescale_min_max(X[i])
    if part_val:
        x_train, x_val, x_test = split_training_set(X, part_train=part_train, part_test=part_test, part_val=part_val, seed=seed, reshape=reshape)
        y_train, y_val, y_test = ssplit_training_set(Y, part_train=part_train, part_test=part_test, part_val=part_val, seed=seed, reshape=reshape)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        x_train, x_test = split_training_set(X, part_train=part_train, part_test=part_test, part_val=part_val, seed=seed, reshape=reshape)
        y_train, y_test = ssplit_training_set(Y, part_train=part_train, part_test=part_test, part_val=part_val, seed=seed, reshape=reshape)
        return, x_train, x_test, y_train, y_test


def split_training_set(total_set, part_train=0.8, part_test=0.2, part_val=None, seed=4324, reshape=True):
    ntotal = total_set.shape[0]
    npix = total_set.shape[1]
    indx = np.arange(ntotal)
    ntrains = int(ntotal*part_train)
    ntests = int(ntotal*part_test)
    if seed:
        np.random.seed(seed)
    train_indx = numpy.random.choice(indx, ntrains)
    train = total_set[train_indx]
    indx = np.delete(indx, train_indx)
    test_indx = numpy.random.choice(indx, ntest)
    test = total_set[test_indx]
    if part_val:
        val_indx = np.delete(indx, train_index)
        nval = len(val_indx)
        val = total_set[val_indx]
    if reshape:
        train = train.reshape(ntrains, npix, npix, 1)
        test = test.reshapwe(ntest, npix, npix, 1)
        if part_val:
            val = val.reshapwe(nval, npix, npix, 1)
    if part_val:
        return train, val, test
    else:
        return train, val

def rescale_min_max(img, a=-1, b=1):
    img_resc = (b-a)*(img-np.min(img)/(np.max(img)-np.max(img))+a
    return img_resc
