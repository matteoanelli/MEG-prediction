import os
from mne import filter as fl
import sys
import numpy as np

import scipy.io as sio


def import_data(datadir, filename):
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        dataset = sio.loadmat(os.path.join(data_dir, file_name))
        X = dataset['train_data'].astype(np.float)
        y = dataset['train_dg'][:, 0] # considering only the thumb

        print('input data shape: X: {}, y: {}'.format(X.shape, y.shape))

        return X, y
    else:
        print("No such file '{}'".format(path), file=sys.stderr)

def filter_data(X):
    band_ranges = [(1, 60), (60, 100), (100, 200)]

    X_filtered = np.zeros((X.shape[0], X.shape[1] * len(band_ranges)), dtype=float)
    for index, band in enumerate(band_ranges):
        print('range: {} , {}'.format(X.shape[1]*index, X.shape[1]*index+1))
        X_filtered[:, X.shape[1]*index:X.shape[1]*index+1] = fl.filter_data(X,500, band[0],band[1], method='fir')

    return  X_filtered

def find_event():
    pass

def create_epoch():
    pass

def y_resampling():
    pass

def pre_process(X, y):
    pass


data_dir  = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
file_name = 'sub1_comp.mat'

X, y = import_data(data_dir, file_name)

X_new = filter_data(X)

print(X.shape) # 400000 time samples x 62 channel
print(y.shape) # 400000 time samples x 5 fingers

print(X_new.shape)
print('Example of fingers position : {}'.format(y[0]))








