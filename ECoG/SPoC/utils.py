import errno
import os
import pickle
import sys

import mne
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from mne.decoding import Scaler

sys.path.insert(1, r"")

from MEG.Utils.utils import *


def import_ECoG(
    datadir,
    filename,
    finger,
    duration,
    overlap,
    normalize_input,
    y_measure="movement",
):
    # TODO add finger choice dict
    path = "".join([datadir, filename])
    if os.path.exists(path):
        dataset = sio.loadmat(path)
        X = dataset["train_data"].astype(np.float).T
        assert (
            finger >= 0 and finger < 5
        ), "Finger input not valid, range value from 0 to 4."
        y = dataset["train_dg"][:, finger]  #

        raw = create_raw(X, X.shape[0], sampling_rate=1000)
        # Generate fixed length events.
        events = mne.make_fixed_length_events(
            raw, duration=duration, overlap=overlap
        )
        # Notch filter out some specific noisy bands
        raw.notch_filter([50, 100])
        # Band pass the input data
        raw.filter(l_freq=1.0, h_freq=70)

        epochs = mne.Epochs(
            raw, events, tmin=0.0, tmax=duration, baseline=(0, 0), decim=2
        )

        X = epochs.get_data()

        bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
        bp = bandpower_multi(
            X, fs=epochs.info["sfreq"], bands=bands, relative=True
        )

        # Normalize data
        if normalize_input:
            X = standard_scaling(X, scalings="mean", log=True)

        # Pick the y vales per each hand
        y = y_reshape(y_PCA(y), measure=y_measure)

        print(
            "The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}".format(
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
    # band_ranges = [(100, 200)]
    X_filtered = np.zeros(
        (X.shape[0], X.shape[1] * len(band_ranges)), dtype=float
    )
    for index, band in enumerate(band_ranges):
        X_filtered[
            :, X.shape[1] * index : X.shape[1] * (index + 1)
        ] = mne.filter.filter_data(
            X, sampling_rate, band[0], band[1], method="fir"
        )
    mne.filter.notch_filter(X_filtered, sampling_rate, [50, 100])

    return X_filtered


def find_events(raw, duration=5.0, overlap=1.0):
    events = mne.make_fixed_length_events(
        raw, duration=duration, overlap=overlap
    )
    return events


def create_raw(X, n_channels, sampling_rate):

    info = mne.create_info(n_channels, sampling_rate, "ecog")
    info["description"] = "ECoG dataset IV BCI competition"

    return mne.io.RawArray(X, info)


def create_epoch(
    X,
    sampling_rate,
    duration=4.0,
    overlap=0.0,
    ds_factor=1.0,
    verbose=None,
    baseline=None,
):
    # Create Basic info data
    # X.shape to be channel, n_sample
    n_channels = X.shape[0]
    raw = create_raw(X, n_channels, sampling_rate)

    # events = mne.make_fixed_length_events(raw, 1, duration=duration)
    # delta = 1. / raw.info['sfreq'] # TODO understand this delta
    # epochs = mne.Epochs(raw, events, event_id=[1], tmin=tmin,
    #               tmax=tmax - delta,
    #               verbose=verbose, baseline=baseline)

    events = mne.make_fixed_length_events(
        raw, 1, duration=duration, overlap=overlap
    )
    delta = 1.0 / raw.info["sfreq"]

    if float(ds_factor) != 1.0:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=[1],
            tmin=0.0,
            tmax=duration - delta,
            verbose=verbose,
            baseline=baseline,
            preload=True,
        )
        epochs = epochs.copy().resample(sampling_rate / ds_factor, npad="auto")
    else:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=[1],
            tmin=0.0,
            tmax=duration - delta,
            verbose=verbose,
            baseline=baseline,
            preload=False,
        )

    return epochs


def standard_scaling(data, scalings="mean", log=False):

    if log:
        data = np.log(data + np.finfo(np.float32).eps)

    if scalings in ["mean", "median"]:
        scaler = Scaler(scalings=scalings)
        data = scaler.fit_transform(data)
    else:
        raise ValueError("scalings should be mean or median")

    return data


def y_resampling(y, n_chunks, scaling=True):

    split = np.array_split(y, n_chunks)

    # y = np.sum(np.abs(split), axis=-1)
    y = np.expand_dims(np.array([np.sum(np.abs(c)) for c in split]), axis=1)
    if scaling:
        y = standard_scaling(y, log=False)
    return y.squeeze()


def split_data(X, y, test_size=0.3, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def normalize(X, y):
    # print(X.shape)
    # print(y.shape)
    #
    # scaler_X = MinMaxScaler()
    # scaler_y = MinMaxScaler()
    #
    # print('Start scaling')
    # scaler_X.fit_transform(X)
    # scaler_y.fit_transform(y)
    pass


def pre_process(X, y):
    pass


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


# TODO fix all the Transpose function coherently
# TODO Save and Load model
