import errno
import os
import pickle
import sys

import numpy as np
import mne
from sklearn.model_selection import train_test_split
import torch

def window_stack(x, window, overlap, sample_rate):
    window_size = round(window * sample_rate)
    stride = round((window - overlap) * sample_rate)
    print(x.shape)
    print("window {}, stride {}, x.shape {}".format(window_size, stride, x.shape))

    return torch.cat([x[:, i : min(x.shape[1], i + window_size)] for i in range(0, x.shape[1], stride)], dim=1,)


def import_MEG(raw_fnames, duration, overlap):
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

    y_left = y_reshape(epochs.get_data()[:, accelerometer_picks_left, :])
    y_right = y_reshape(epochs.get_data()[:, accelerometer_picks_right, :])

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {},"\
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )
    return X, y_left, y_right

def import_MEG_Tensor(raw_fnames, duration, overlap):

    X, y_left, y_right = import_MEG(raw_fnames, duration, overlap)

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

def y_reshape(y):
    # the y has 2 position
    y = np.sqrt(np.mean(np.power(y[:, 0, :], 2), axis=1))
    return y

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

# TODO add notch filter