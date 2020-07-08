import os
import mne
import sys
import numpy as np
import sklearn

from mne.decoding import SPoC as SPoc
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import scipy.io as sio
import matplotlib.pyplot as plt

# %%
def import_data(datadir, filename, finger):
    # TODO add finger choice dict
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(data_dir, file_name))
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

#%%
# data_dir  = os.environ['DATA_PATH']
data_dir = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
file_name = 'sub1_comp.mat'
sampling_rate = 1000

#%%
# Example
X, y = import_data(data_dir, file_name, 0)

print('Example of fingers position : {}'.format(y[0]))
print('epochs with events generation')
epochs = create_epoch(X, sampling_rate, 4., 1)


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
print('X shape {}, y shape {}'.format(X.shape, y.shape))
spoc_estimator = SPoc(n_components=10, log=True, reg='oas', rank='full')

X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

print('X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

pipeline = make_pipeline(spoc, Ridge())

# X_new = spoc_estimator.fit_transform(X_train, y_train)
# regressor = Ridge()
# regressor.fit(X_new, y_train)

# y_new = regressor.predict(spoc_estimator.transform(X_test))

pipeline.fit(X_train, y_train)

y_new_train = pipeline.predict(X_train)
y_new = pipeline.predict(X_test)

print('mean squared error {}'.format(mean_squared_error(y_test, y_new)))
print('mean absolute error {}'.format(mean_absolute_error(y_test, y_new)))

# plot y_new against the true value
fig, ax = plt.subplots(1, 2, figsize=[10, 4])
times = np.arange(y_new.shape[0])
ax[0].plot(times, y_new, color='b', label='Predicted')
ax[0].plot(times, y_test, color='r', label='True')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Finger Movement')
ax[0].set_title('SPoC Finger Movement')
times = np.arange(y_new_train.shape[0])
ax[1].plot(times, y_new_train, color='b', label='Predicted')
ax[1].plot(times, y_train, color='r', label='True')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Finger Movement')
ax[1].set_title('SPoC Finger Movement training')
plt.legend()
mne.viz.tight_layout()
plt.show()

# y_pred = cross_val_predict(pipeline, X, y, cv=cv)

# print(mean_squared_error(y, y_pred))


# TODO,  find a way to evaluate the pipeline as well as the SPoC algorithm done
# TODO, implement approach not using epoched data (form continuous data)
# TODO, implement normalization
# TODO, implement classical split test done
# TODO, channel selection




