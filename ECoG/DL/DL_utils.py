import os
import scipy.io as sio
import sys

import numpy as np
import torch

def import_ECoG_Tensor(datadir, filename, finger, window_size, sample_rate):
    # TODO add finger choice dict
    # TODO refactor reshaping of X (add the mean)
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(datadir, filename))
        X = torch.from_numpy(dataset['train_data'].astype(np.float32).T)

        module = X.shape[1]%int(window_size*sample_rate)
        if module != 0:
            X = X[:, :-module] # Discard some of the last time points to allow the reshape
        X = torch.reshape(X, (-1, X.shape[0], int(window_size*sample_rate)))
        assert finger >= 0 and finger <5, 'Finger input not valid, range value from 0 to 4.'
        y = torch.from_numpy(dataset['train_dg'][:, finger].astype(np.float32))

        if module != 0:
            y = y[:-module]
        y = y_resampling(y, X.shape[2])

        print('The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}'
              .format(X.shape, y.shape))

        return X, y
    else:
        print("No such file '{}'".format(path), file=sys.stderr)
        return None, None

def y_resampling(y, n_chunks):

    split = list(torch.split(y, n_chunks))
    y = torch.stack([torch.mean(s) for s in split])

    return y