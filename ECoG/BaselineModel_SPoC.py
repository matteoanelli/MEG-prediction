import os
import mne
import sys
import numpy as np
import sklearn

from mne.decoding import SPoC as SPoc
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

import scipy.io as sio
import matplotlib.pyplot as plt

# %%
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

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


def pre_process(X, y):
    pass

#%%
data_dir  = os.environ['DATA_PATH']
file_name = 'sub1_comp.mat'
sampling_rate = 1000

#%%
# Example
X, y = import_data(data_dir, file_name)

print('Example of fingers position : {}'.format(y[0]))


# find event on Y target
print('epochs without events generation')
epochs = create_epoch(X, sampling_rate)
print(epochs)

print('epochs with events generation')
epochs_event = create_epoch(X, sampling_rate, True)
print(epochs_event)

# X = epochs_event.get_data()
X = epochs.get_data()
#%%

y = y_resampling(y, X.shape[0])
#%%
# SPoC algotithms

spoc = SPoc(n_components=2, log=True, reg='oas', rank='full')

cv = KFold(n_splits=4, shuffle=False)

pipeline = make_pipeline(spoc, Ridge())

scores_1 = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')

pipeline2 = make_pipeline(spoc, Ridge())


scores = cross_validate(pipeline2, X, y, cv=cv, scoring='neg_root_mean_squared_error', return_estimator=True)

print('Cross_val_score score : {}'.format(scores_1))
print('Cross_validate score : {}'.format(scores))

print(pipeline.get_params())
# %%

spoc_estimator = SPoc(n_components=2, log=True, reg='oas', rank='full')
spoc_estimator.fit(X, y)
spoc_estimator.plot_patterns(epochs.info)

print(spoc_estimator.get_params())

# Run cross validaton
# y_pred = cross_val_predict(pipeline, X, y, cv=cv)

# print(mean_squared_error(y, y_pred))


# TODO,  find a way to evaluate the pipeline as well as the SPoC algorithm
# TODO, implement approach not using epoched data (form continuous data)
# TODO, implement normalization
# TODO, implement classical split test




