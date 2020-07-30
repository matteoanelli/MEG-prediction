import os
import mne
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
import pickle
import errno


import scipy.io as sio

def import_ECoG(datadir, filename, finger):
    # TODO add finger choice dict
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(datadir, filename))
        X = dataset['train_data'].astype(np.float).T
        assert finger >= 0 and finger <5, 'Finger input not valid, range value from 0 to 4.'
        y = dataset['train_dg'][:, finger] #

        print('The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}'
              .format(X.shape, y.shape))
        return X, y
    else:
        print("No such file '{}'".format(path), file=sys.stderr)

def filter_data(X, sampling_rate):
    # TODO appropriate filtering and generalize function

    # careful with x shape, the last dimension should be n_times
    band_ranges = [(60, 200)]
    # band_ranges = [(100, 200)]
    X_filtered = np.zeros((X.shape[0], X.shape[1] * len(band_ranges)), dtype=float)
    for index, band in enumerate(band_ranges):
        X_filtered[:, X.shape[1]*index:X.shape[1]*(index+1)] = mne.filter.filter_data(X,sampling_rate, band[0],band[1], method='fir')

    return  X_filtered

def find_events(raw,duration=5., overlap=1.):
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    return events

def create_raw(X, n_channels, sampling_rate):


    info = mne.create_info(n_channels, sampling_rate, 'ecog')
    info['description'] = 'ECoG dataset IV BCI competition'

    return mne.io.RawArray(X, info)

def create_epoch(X, sampling_rate, duration=4., overlap=0., ds_factor=1., verbose=None, baseline=None):
    # Create Basic info data
    # X.shape to be channel, n_sample
    n_channels = X.shape[0]
    raw = create_raw(X, n_channels, sampling_rate)


    # events = mne.make_fixed_length_events(raw, 1, duration=duration)
    # delta = 1. / raw.info['sfreq'] # TODO understand this delta
    # epochs = mne.Epochs(raw, events, event_id=[1], tmin=tmin,
    #               tmax=tmax - delta,
    #               verbose=verbose, baseline=baseline)

    events = mne.make_fixed_length_events(raw, 1, duration=duration, overlap=overlap)
    delta = 1. / raw.info['sfreq']

    if float(ds_factor) != 1.:
        epochs = mne.Epochs(raw, events, event_id=[1], tmin=0.,
                            tmax=duration - delta,
                            verbose=verbose, baseline=baseline, preload=True)
        epochs = epochs.copy().resample(sampling_rate/ds_factor, npad='auto')
    else:
        epochs = mne.Epochs(raw, events, event_id=[1], tmin=0.,
                            tmax=duration - delta,
                            verbose=verbose, baseline=baseline, preload=False)

    return epochs

def y_resampling(y, n_chunks):
    # TODO find a way to process the y data respect to the events (discretize the y)
    # Continuous data down sampling
    # Simply aggregate data based on number of epochs
    split = np.array_split(y, n_chunks)
    y = np.array([np.mean(c) for c in split])

    return y

def split_data(X, y, test_size=0.4, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def normalize(X, y):
    print(X.shape)
    print(y.shape)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    print('Start scaling')
    scaler_X.fit_transform(X)
    scaler_y.fit_transform(y)

    return scaler_X.fit_transform(X), scaler_y.fit_transform(y)

def pre_process(X, y):
    pass

def save_skl_model(esitimator, models_path, name):
    if os.path.exists(models_path):
        pickle.dump(esitimator, open(os.path.join(models_path, name), 'wb'))
        print('Model saved successfully.')
    else:
        FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), models_path)


def load_skl_model(models_path):
    with open(models_path, 'rb') as model:
        model = pickle.load(model)
        print('Model loaded successfully.')
        return model


# TODO fix all the Transpose function coherently
# TODO Save and Load model
