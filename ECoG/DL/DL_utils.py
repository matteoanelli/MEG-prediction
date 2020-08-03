import os
import scipy.io as sio
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import mne


def window_stack(x, window, overlap, sample_rate):
    window_size = round(window*sample_rate)
    stride = round((window-overlap)*sample_rate)
    print(x.shape)
    print('window {}, stride {}, x.shape {}'.format(window_size, stride, x.shape))

    return torch.cat([x[:, i:min(x.shape[1], i+window_size)] for i in range(0, x.shape[1], stride)], dim=1)
    # return [x[:, i:min(x.shape[1], i+window_size)] for i in range(0, x.shape[1], stride)]


def import_ECoG_Tensor(datadir, filename, finger, window_size=0.5, sample_rate=1000, overlap=0.):
    # TODO add finger choice dict
    # TODO refactor reshaping of X (for instance add the mean)
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(datadir, filename))
        X = torch.from_numpy(dataset['train_data'].astype(np.float32).T)
        y = torch.from_numpy(dataset['train_dg'][:, finger].astype(np.float32))

        if overlap != 0.:
            X = window_stack(X, 0.5, 0.25, sample_rate)
            y = window_stack(y.unsqueeze(0), 0.5, 0.25, sample_rate).squeeze()

        module = X.shape[1] % int(window_size*sample_rate)
        if module != 0:
            X = X[:, :-module] # Discard some of the last time points to allow the reshape
        X = torch.reshape(X, (-1, X.shape[0], int(window_size*sample_rate)))
        assert 0 <= finger < 5, 'Finger input not valid, range value from 0 to 4.'

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


def save_pytorch_model(model, path, filename):

    if os.path.exists(path):
        do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
        if do_save == 'yes' or do_save == 'y':
            torch.save(model.state_dict(), os.path.join(path, filename))
            print('Model saved to {}.'.format(os.path.join(path, filename)))
        else:
            print('Model not saved.')
    else:
        raise Exception('The path does not exist, path: {}'.format(path))


def load_pytorch_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    print('Model loaded from {}.'.format(path))
    model.to(device)
    model.eval()
    return model


def split_data(X, y, test_size=0.4, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def downsampling(x, down=1, npad=100, axis=-1, window='boxcar', n_jobs=1, pad='reflect_limited', verbose=None):
    # TODO check padding and window choices

    return torch.from_numpy(mne.filter.resample(x.numpy().astype('float64'), down=down, npad=npad,
                                                axis=axis, window=window, n_jobs=n_jobs, pad=pad, verbose=verbose))


def filtering():

    pass
