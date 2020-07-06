import os
import mne
import sys
import numpy as np
import sklearn

from mne.decoding import SPoC as SPoc
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict


import scipy.io as sio


def import_data(datadir, filename):
    # TODO add finger choice
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(data_dir, file_name))
        X = dataset['train_data'].astype(np.float)
        y = dataset['train_dg'][:, 0] # considering only the thumb

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

def create_epoch(X, sampling_rate, events=False, duration=4.):
    # TODO implement possibility of sliding window
    # Create Basic info data
    n_channels = X.shape[1]
    raw = create_raw(X, n_channels, sampling_rate)
    if events is False:
        epochs = mne.make_fixed_length_epochs(raw, duration)
    else:
        epochs = mne.Epochs(raw, find_events(raw))
    return epochs

def y_resampling(y, n_chunks):
    # TODO find a way to process the y data respect to the events (discretize the y)
    # Simply aggregate data based on number of epochs
    split = np.array_split(y, n_chunks)
    y = np.array([np.mean(c) for c in split])

    return y


def pre_process(X, y):
    pass


data_dir  = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
file_name = 'sub1_comp.mat'
sampling_rate = 1000

X, y = import_data(data_dir, file_name)

print('Example of fingers position : {}'.format(y[0]))


# find event on Y target
print('epochs without events generation')
epochs = create_epoch(X, sampling_rate)
print(epochs)

print('epochs with events generation')
epochs_event = create_epoch(X, sampling_rate, True)
print(epochs_event)

X = epochs_event.get_data()
y = y_resampling(y, X.shape[0])

# SPoC algotithms

spoc = SPoc(n_components=2, log=True, reg='oas', rank='full')







