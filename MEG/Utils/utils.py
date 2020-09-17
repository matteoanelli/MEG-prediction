import errno
import os
import pickle
import sys

import mne
import numpy as np
import torch
from mne.decoding import Scaler
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from numpy import trapz
from scipy.integrate import cumtrapz

def window_stack(x, window, overlap, sample_rate):
    window_size = round(window * sample_rate)
    stride = round((window - overlap) * sample_rate)
    print(x.shape)
    print("window {}, stride {}, x.shape {}".format(window_size, stride, x.shape))

    return torch.cat([x[:, i: min(x.shape[1], i + window_size)] for i in range(0, x.shape[1], stride)], dim=1,)


def import_MEG(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement"):
    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            raw = mne.io.Raw(raw_fnames[0], preload=True)
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
            raw.pick_types(meg='grad', misc=True)
            raw.notch_filter([50, 100])
            raw.filter(l_freq=1., h_freq=70)

            # get indices of accelerometer channels
            accelerometer_picks_left = mne.pick_channels(raw.info['ch_names'],
                                                         include=["MISC001", "MISC002"])
            accelerometer_picks_right = mne.pick_channels(raw.info['ch_names'],
                                                          include=["MISC003", "MISC004"])
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0)))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    epochs = mne.concatenate_epochs(epochs)
    # get indices of accelerometer channels

    # pic only with gradiometer
    X = epochs.get_data()[:, :204, :]

    if normalize_input:
        X = standard_scaling(X, scalings="mean", log=True)

    y_left = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_left, :]), measure=y_measure)
    y_right = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_right, :]), measure=y_measure)

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {},"\
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )
    return X, y_left, y_right


def import_MEG_Tensor(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement"):

    X, y_left, y_right = import_MEG(raw_fnames, duration, overlap, normalize_input=normalize_input, y_measure=y_measure)

    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

    y_left = torch.from_numpy(y_left.astype(np.float32))
    y_right = torch.from_numpy(y_right.astype(np.float32))

    return X, torch.stack([y_left, y_right], dim=1)

def import_MEG_Tensor_form_file(data_dir, normalize_input=True, y_measure="movement"):

    print("Using saved epoched data, loading...")
    X = np.fromfile(os.path.join(data_dir, "X.dat"), dtype=float)
    y_left = np.fromfile(os.path.join(data_dir, "y_left.dat"), dtype=float)
    y_right = np.fromfile(os.path.join(data_dir, "y_right.dat"), dtype=float)
    print("Data loaded!")

    if normalize_input:
        X = standard_scaling(X, scalings="mean", log=True)

    y_left = y_reshape(y_PCA(y_left), measure=y_measure)
    y_right = y_reshape(y_PCA(y_right), measure=y_measure)

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {},"\
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )

    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

    y_left = torch.from_numpy(y_left.astype(np.float32))
    y_right = torch.from_numpy(y_right.astype(np.float32))


    return X, torch.stack([y_left, y_right], dim=1)

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


def y_reshape(y, measure="mean", scaling=True):
    # the y has 2 position
    if measure == 'mean':
        y = np.sqrt(np.mean(np.power(y, 2), axis=-1))

    elif measure == 'movement':
        y = np.sum(np.abs(y), axis=-1)
        if scaling:
            y = standard_scaling(y, log=False)

    elif measure == 'velocity':
        y = trapz(y, axis=-1)/y.shape[-1]
        if scaling:
            y = standard_scaling(y, log=False)

    elif measure == 'position':
        vel = cumtrapz(y, axis=-1)
        y = trapz(vel, axis=-1)/y.shape[-1]
        if scaling:
            y = standard_scaling(y, log=False)

    else:
        raise ValueError("measure should be one of: mean, movement, velocity, position")

    return y.squeeze()


def y_PCA(y):

    pca = UnsupervisedSpatialFilter(PCA(1), average=False)

    return pca.fit_transform(y)


def save_pytorch_model(model, path, filename):

    if os.path.exists(path):
        # do_save = input("Do you want to save the model (type yes to confirm)? ").lower()
        do_save = 'y'
        if do_save == "yes" or do_save == "y":
            torch.save(model.state_dict(), os.path.join(path, filename))
            print("Model saved to {}.".format(os.path.join(path, filename)))
        else:
            print("Model not saved.")
    else:
        raise Exception("The path does not exist, path: {}".format(path))


def load_pytorch_model(model, path, device):
    # model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(path))
    print("Model loaded from {}.".format(path))
    model.to(device)
    model.eval()
    return model


def normalize(data):

    # linear rescale to range [0, 1]
    min = torch.min(data.view(data.shape[2], -1), dim=1, keepdim=True)[0]
    data -= min.view(1, 1, min.shape[0], 1)
    max = torch.max(data.view(data.shape[2], -1), dim=1, keepdim=True)[0]
    data /= max.view(1, 1, max.shape[0], 1)

    # Linear rescale to range [-1, 1]
    return 2 * data - 1

def standard_scaling(data, scalings="mean", log=True):

    if log:
        data = np.log(data + np.finfo(np.float32).eps)

    if scalings in ["mean", "median"]:
        scaler = Scaler(scalings=scalings)
        data = scaler.fit_transform(data)
    else:
        raise ValueError("scalings should be mean or median")

    return data

def transform_data():
    pass

def len_split(len):

    if len * 0.7 - int(len*0.7) == 0. and len * 0.15 - int(len*0.15) >= 0.:
        train = round(len * 0.7)
        valid = round(len * 0.15)
        test = round(len * 0.15)

    elif len * 0.7 - int(len*0.7) >= 0.5:
        if len * 0.15 - int(len*0.15) >= 0.5:
            train = round(len * 0.7)
            valid = round(len * 0.15)
            test = round(len * 0.15) - 1
        else:
            train = round(len * 0.7)
            valid = round(len * 0.15)
            test = round(len * 0.15)

    else:
        if len * 0.15 - int(len*0.15) >= 0.5:
            train = round(len * 0.7)
            valid = round(len * 0.15)
            test = round(len * 0.15)
        else:
            train = round(len * 0.7)
            valid = round(len * 0.15) + 1
            test = round(len * 0.15)

    return train, valid, test





# TODO add notch filter