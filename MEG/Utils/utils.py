import errno
import os
import pickle
import sys

import numpy as np
from mne import read_epochs, filter
from sklearn.model_selection import train_test_split


def import_MEG(datadir, filename):
    path = os.path.join(datadir, filename)
    if os.path.exists(path):
        # Import pre-epoched data
        X = np.array(read_epochs(os.path.join(path)).get_data())
        # TODO insert actual Y
        t = np.linspace(0, 1, 180, endpoint=False)
        y = np.sin(30 * np.pi * t)

        print(
            "The input data are of shape: {}, the corresponding y shape is: {}".format(
                X.shape, y.shape
            )
        )
        return X, y
    else:
        print("No such file '{}'".format(path), file=sys.stderr)

def filter_data(X, sampling_rate):
    # TODO appropriate filtering and generalize function

    # careful with x shape, the last dimension should be n_times
    band_ranges = [(60, 200)]
    X_filtered = np.zeros((X.shape[0], X.shape[1] * len(band_ranges)), dtype=float)
    for index, band in enumerate(band_ranges):
        X_filtered[:, X.shape[1] * index : X.shape[1] * (index + 1)] = filter.filter_data(
            X, sampling_rate, band[0], band[1], method="fir"
        )

    return X_filtered

def split_data(X, y, test_size=0.3, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def save_skl_model(esitimator, models_path, name):
    if os.path.exists(models_path):
        pickle.dump(esitimator, open(os.path.join(models_path, name), "wb"))
        print("Model saved successfully.")
    else:
        FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), models_path)


def load_skl_model(models_path):
    with open(models_path, "rb") as model:
        model = pickle.load(model)
        print("Model loaded successfully.")
        return model
