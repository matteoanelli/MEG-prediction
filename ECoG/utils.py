import os
import mne
import sys
import numpy as np

from sklearn.model_selection import train_test_split


import scipy.io as sio

def import_data(datadir, filename, finger):
    # TODO add finger choice dict
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(datadir, filename))
        X = dataset['train_data'].astype(np.float)
        assert finger >= 0 and finger <5, 'Finger input not valid, range value from 0 to 4.'
        y = dataset['train_dg'][:, finger] #

        print('The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}'.format(X.shape,
                                                                                                                y.shape))
        return X, y
    else:
        print("No such file '{}'".format(path), file=sys.stderr)

def filter_data(X):
    # TODO appropriate filtering
    band_ranges = [(1, 60), (60, 100), (100, 200)]

    X_filtered = np.zeros((X.shape[0], X.shape[1] * len(band_ranges)), dtype=float)
    for index, band in enumerate(band_ranges):
        print('range: {} , {}'.format(X.shape[1]*index, X.shape[1]*index+1))
        X_filtered[:, X.shape[1]*index:X.shape[1]*index+1] = mne.filter.filter_data(X,500, band[0],band[1], method='fir')

    return  X_filtered

def find_events(raw,duration=5., overlap=1.):
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    return events

def create_raw(X, n_channels, sampling_rate):


    info = mne.create_info(n_channels, sampling_rate, 'ecog')
    info['description'] = 'ECoG dataset IV BCI competition'

    return mne.io.RawArray(X.T, info)

def create_epoch(X, sampling_rate, duration=4., overlap=0., verbose=None, baseline=None):
    # Create Basic info data
    n_channels = X.shape[1]
    raw = create_raw(X, n_channels, sampling_rate)

    # events = mne.make_fixed_length_events(raw, 1, duration=duration)
    # delta = 1. / raw.info['sfreq'] # TODO understand this delta
    # epochs = mne.Epochs(raw, events, event_id=[1], tmin=tmin,
    #               tmax=tmax - delta,
    #               verbose=verbose, baseline=baseline)
    events = mne.make_fixed_length_events(raw, 1, duration=duration, overlap=overlap)
    delta = 1. / raw.info['sfreq']
    epochs = mne.Epochs(raw, events, event_id=[1], tmin=0.,
                        tmax=duration - delta,
                        verbose=verbose, baseline=baseline)
    return epochs

def y_resampling(y, n_chunks):
    # TODO find a way to process the y data respect to the events (discretize the y)
    # Simply aggregate data based on number of epochs
    split = np.array_split(y, n_chunks)
    y = np.array([np.mean(c) for c in split])

    return y

def split_data(X, y, test_size=0.4, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def pre_process(X, y):
    pass
