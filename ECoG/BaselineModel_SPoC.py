#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""

from utils import *

from mne.decoding import SPoC as SPoc
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt


#%%
# data_dir  = os.environ['DATA_PATH']
data_dir = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
file_name = 'sub1_comp.mat'
sampling_rate = 1000

#%%
# Example
X, y = import_ECoG(data_dir, file_name, 0)
print(X.shape)
X = filter_data(X, sampling_rate)
print(X.shape)
print('Example of fingers position : {}'.format(y[0]))
print('epochs with events generation')
epochs = create_epoch(X, sampling_rate, duration=4., overlap=1., ds_factor=1)


# X = epochs_event.get_data()
X = epochs.get_data()
#%%

y = y_resampling(y, X.shape[0])
# X, y = normalize(X, y)
# SPoC algotithms

# spoc = SPoc(n_components=2, log=True, reg='oas', rank='full')
#
# cv = KFold(n_splits=4, shuffle=False)
#
# pipeline = make_pipeline(spoc, Ridge())
#
# scores_1 = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
#
# pipeline2 = make_pipeline(spoc, Ridge())
#
#
# scores = cross_validate(pipeline2, X, y, cv=cv, scoring='neg_root_mean_squared_error', return_estimator=True)
#
# print('Cross_val_score score : {}'.format(scores_1))
# print('Cross_validate score : {}'.format(scores))
#
# print(pipeline.get_params())
# %%
print('X shape {}, y shape {}'.format(X.shape, y.shape))
spoc_estimator = SPoc(n_components=2, log=True, reg='oas', rank='full')

X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

print('X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

pipeline = make_pipeline(spoc_estimator, Ridge())

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
# fig, ax = plt.subplots(1, 2, figsize=[10, 4])
# times = np.arange(y_new.shape[0])
# ax[0].plot(times, y_new, color='b', label='Predicted')
# ax[0].plot(times, y_test, color='r', label='True')
# ax[0].set_xlabel('Epoch')
# ax[0].set_ylabel('Finger Movement')
# ax[0].set_title('SPoC Finger Movement')
# # times = np.arange(y_new_train.shape[0])
# # ax[1].plot(times, y_new_train, color='b', label='Predicted')
# # ax[1].plot(times, y_train, color='r', label='True')
# # ax[1].set_xlabel('Epoch')
# # ax[1].set_ylabel('Finger Movement')
# # ax[1].set_title('SPoC Finger Movement training')
# # plt.legend()
# mne.viz.tight_layout()
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = np.arange(y_new.shape[0])
ax.plot(times, y_new, color='b', label='Predicted')
ax.plot(times, y_test, color='r', label='True')
ax.set_xlabel('Epoch')
ax.set_ylabel('Finger Movement')
ax.set_title('SPoC Finger Movement')
plt.legend()
mne.viz.tight_layout()
plt.show()


# y_pred = cross_val_predict(pipeline, X, y, cv=cv)

# print(mean_squared_error(y, y_pred))


# TODO, implement approach not using epoched data (form continuous data)
# TODO, implement normalization
# TODO, find a good way to create the epochs. In term of window duration
# TODO, Implement of y resampling
# TODO, Implement filters
# TODO, component analysis





