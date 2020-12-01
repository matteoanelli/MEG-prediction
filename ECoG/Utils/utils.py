import errno
import os
import pickle
import sys

import mne
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from mne.decoding import Scaler

sys.path.insert(1, r'')

from MEG.Utils.utils import *


def create_raw(X, y, n_channels, sampling_rate):

    info = mne.create_info(n_channels+1, sampling_rate, "ecog")
    info["description"] = "ECoG dataset IV BCI competition"

    return mne.io.RawArray(np.concatenate((X, np.expand_dims(y, axis=0)), axis=0), info)


def import_ECoG(datadir, filename, finger, duration, overlap, normalize_input=True, y_measure="mean"):
    # TODO add finger choice dict
    path = "".join([datadir, filename])
    if os.path.exists(path):
        dataset = sio.loadmat(path)
        X = dataset["train_data"].astype(np.float).T
        assert finger >= 0 and finger < 5, "Finger input not valid, range value from 0 to 4."
        y = dataset["train_dg"][:, finger]  #

        raw = create_raw(X, y, X.shape[0], sampling_rate=1000)

        # Generate fixed length events.
        events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
        # Notch filter out some specific noisy bands
        raw.notch_filter([50, 100])
        # Band pass the input data
        raw.filter(l_freq=1., h_freq=70)

        epochs = mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0), decim=2)

        X = epochs.get_data()[:, :-1, :]
        y = epochs.get_data()[:, -1, :]

        # bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
        # bp = bandpower_multi(X, fs=epochs.info['sfreq'], bands=bands, relative=True)

        # Normalize data
        if normalize_input:
            X = standard_scaling(X, scalings="mean", log=False)

        # Pick the y vales per each hand
        y = y_reshape(np.expand_dims(y, axis=1), measure=y_measure)

        print(
            "The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}".format(
                X.shape, y.shape
            )
        )
        return X, y
    else:
        print("No such file '{}'".format(path), file=sys.stderr)


def import_ECoG_rps(datadir, filename, finger, duration, overlap, normalize_input=True, y_measure="mean"):
    # TODO add finger choice dict
    path = "".join([datadir, filename])
    if os.path.exists(path):
        dataset = sio.loadmat(path)
        X = dataset["train_data"].astype(np.float).T
        assert finger >= 0 and finger < 5, "Finger input not valid, range value from 0 to 4."
        y = dataset["train_dg"][:, finger]  #

        raw = create_raw(X, y, X.shape[0], sampling_rate=1000)

        # Generate fixed length events.
        events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
        # Notch filter out some specific noisy bands
        raw.notch_filter([50, 100])
        # Band pass the input data
        raw.filter(l_freq=1., h_freq=70)

        epochs = mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0), decim=2)

        X = epochs.get_data()[:, :-1, :]
        y = epochs.get_data()[:, -1, :]

        bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
        bp = bandpower_multi(X, fs=epochs.info['sfreq'], bands=bands, relative=True)

        # Normalize data
        if normalize_input:
            X = standard_scaling(X, scalings="mean", log=False)

        # Pick the y vales per each hand
        y = y_reshape(np.expand_dims(y, axis=1), measure=y_measure)

        print(
            "The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}".format(
                X.shape, y.shape
            )
        )
        return X, y, bp
    else:
        print("No such file '{}'".format(path), file=sys.stderr)


def import_ECoG_Tensor(datadir, filename, finger, duration, sample_rate=1000, overlap=0.0, rps=True):

    if rps:
        X, y, bp = import_ECoG_rps(datadir, filename, finger, duration, overlap=overlap, normalize_input=True,
                                   y_measure="mean")
    else:
        X, y = import_ECoG(datadir, filename, finger, duration, overlap=overlap, normalize_input=True,
                           y_measure="mean")

    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

    y = torch.from_numpy(y.astype(np.float32))
    print(y)

    if rps:
        bp = torch.from_numpy(bp.astype(np.float32))
        return X, y, bp
    else:
        return X, y

#
# def filter_data(X, sampling_rate):
#     # TODO appropriate filtering and generalize function
#
#     # careful with x shape, the last dimension should be n_times
#     band_ranges = [(60, 200)]
#     # band_ranges = [(100, 200)]
#     X_filtered = np.zeros((X.shape[0], X.shape[1] * len(band_ranges)), dtype=float)
#     for index, band in enumerate(band_ranges):
#         X_filtered[:, X.shape[1] * index: X.shape[1] * (index + 1)] = mne.filter.filter_data(
#             X, sampling_rate, band[0], band[1], method="fir"
#         )
#     mne.filter.notch_filter(X_filtered, sampling_rate, [50, 100])
#
#     return X_filtered
#
#
# def find_events(raw, duration=5.0, overlap=1.0):
#     events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
#     return events
#
#

#
# def create_epoch(X, sampling_rate, duration=4.0, overlap=0.0, ds_factor=1.0, verbose=None, baseline=None):
#     # Create Basic info data
#     # X.shape to be channel, n_sample
#     n_channels = X.shape[0]
#     raw = create_raw(X, n_channels, sampling_rate)
#
#     # events = mne.make_fixed_length_events(raw, 1, duration=duration)
#     # delta = 1. / raw.info['sfreq'] # TODO understand this delta
#     # epochs = mne.Epochs(raw, events, event_id=[1], tmin=tmin,
#     #               tmax=tmax - delta,
#     #               verbose=verbose, baseline=baseline)
#
#     events = mne.make_fixed_length_events(raw, 1, duration=duration, overlap=overlap)
#     delta = 1.0 / raw.info["sfreq"]
#
#     if float(ds_factor) != 1.0:
#         epochs = mne.Epochs(
#             raw,
#             events,
#             event_id=[1],
#             tmin=0.0,
#             tmax=duration - delta,
#             verbose=verbose,
#             baseline=baseline,
#             preload=True,
#         )
#         epochs = epochs.copy().resample(sampling_rate / ds_factor, npad="auto")
#     else:
#         epochs = mne.Epochs(
#             raw,
#             events,
#             event_id=[1],
#             tmin=0.0,
#             tmax=duration - delta,
#             verbose=verbose,
#             baseline=baseline,
#             preload=False,
#         )
#
#     return epochs
#
#
# def standard_scaling(data, scalings="mean", log=False):
#     if log:
#         data = np.log(data + np.finfo(np.float32).eps)
#
#     if scalings in ["mean", "median"]:
#         scaler = Scaler(scalings=scalings)
#         data = scaler.fit_transform(data)
#     else:
#         raise ValueError("scalings should be mean or median")
#
#     return data
#
#
# def y_resampling(y, n_chunks, scaling=True):
#     split = np.array_split(y, n_chunks)
#
#     # y = np.sum(np.abs(split), axis=-1)
#     y = np.expand_dims(np.array([np.sum(np.abs(c)) for c in split]), axis=1)
#     if scaling:
#         y = standard_scaling(y, log=False)
#     return y.squeeze()
#
#
# def split_data(X, y, test_size=0.3, random_state=0):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#
#     return X_train, X_test, y_train, y_test
#
#
# def normalize(X, y):
#     # print(X.shape)
#     # print(y.shape)
#     #
#     # scaler_X = MinMaxScaler()
#     # scaler_y = MinMaxScaler()
#     #
#     # print('Start scaling')
#     # scaler_X.fit_transform(X)
#     # scaler_y.fit_transform(y)
#     pass
#
#
# def pre_process(X, y):
#     pass
#
#
# def save_skl_model(esitimator, models_path, name):
#     if os.path.exists(models_path):
#         pickle.dump(esitimator, open(os.path.join(models_path, name), "wb"))
#         print("Model saved successfully.")
#     else:
#         FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), models_path)
#
#
# def load_skl_model(models_path):
#     with open(models_path, "rb") as model:
#         model = pickle.load(model)
#         print("Model loaded successfully.")
#         return model
#
# # TODO fix all the Transpose function coherently
# # TODO Save and Load model
#
#
# def window_stack(x, window, overlap, sample_rate):
#     window_size = round(window * sample_rate)
#     stride = round((window - overlap) * sample_rate)
#     print(x.shape)
#     print("window {}, stride {}, x.shape {}".format(window_size, stride, x.shape))
#
#     return torch.cat([x[:, i : min(x.shape[1], i + window_size)] for i in range(0, x.shape[1], stride)], dim=1,)
#     # return [x[:, i:min(x.shape[1], i+window_size)] for i in range(0, x.shape[1], stride)]
#
#
#
#
#
# # def import_ECoG_Tensor(datadir, filename, finger, window_size=0.5, sample_rate=1000, overlap=0.0):
# #     # TODO add finger choice dict
# #     # TODO refactor reshaping of X (for instance add the mean)
# #     path = os.path.join(datadir, filename)
# #     if os.path.exists(path):
# #         dataset = sio.loadmat(os.path.join(datadir, filename))
# #         X = torch.from_numpy(dataset["train_data"].astype(np.float32).T)
# #         y = torch.from_numpy(dataset["train_dg"][:, finger].astype(np.float32))
# #
# #         if overlap != 0.:
# #             X = window_stack(X, window_size, overlap, sample_rate)
# #             y = window_stack(y.unsqueeze(0), window_size, overlap, sample_rate).squeeze()
# #
# #         module = X.shape[1] % int(window_size * sample_rate)
# #         if module != 0:
# #             X = X[:, :-module]  # Discard some of the last time points to allow the reshape
# #         X = torch.reshape(X, (-1, X.shape[0], int(window_size * sample_rate)))
# #         assert 0 <= finger < 5, "Finger input not valid, range value from 0 to 4."
# #
# #         if module != 0:
# #             y = y[:-module]
# #         y = y_resampling(y, X.shape[2])
# #
# #         print(
# #             "The input data are of shape: {}, the corresponding y shape (filtered to 1 finger) is: {}".format(
# #                 X.shape, y.shape
# #             )
# #         )
# #
# #         return X, y
# #     else:
# #         print("No such file '{}'".format(path), file=sys.stderr)
# #         return None, None
#
# def save_pytorch_model(model, path, filename):
#
#     if os.path.exists(path):
#         # do_save = input("Do you want to save the model (type yes to confirm)? ").lower()
#         do_save = "yes"
#         if do_save == "yes" or do_save == "y":
#             torch.save(model.state_dict(), os.path.join(path, filename))
#             print("Model saved to {}.".format(os.path.join(path, filename)))
#         else:
#             print("Model not saved.")
#     else:
#         raise Exception("The path does not exist, path: {}".format(path))
#
#
# def load_pytorch_model(model, path, device):
#     # model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
#     model.load_state_dict(torch.load(path))
#     print("Model loaded from {}.".format(path))
#     model.to(device)
#     model.eval()
#     return model
#
#
# def split_data(X, y, test_size=0.4, random_state=0):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#
#     return X_train, X_test, y_train, y_test
#
#
# def downsampling(
#     x, down=1, npad=100, axis=-1, window="boxcar", n_jobs=1, pad="reflect_limited", verbose=None,
# ):
#     # TODO check padding and window choices
#
#     return torch.from_numpy(
#         filter.resample(
#             x.numpy().astype("float64"),
#             down=down,
#             npad=npad,
#             axis=axis,
#             window=window,
#             n_jobs=n_jobs,
#             pad=pad,
#             verbose=verbose,
#         )
#     )
#
#
# def filtering(
#     x,
#     sfreq,
#     l_freq,
#     h_freq,
#     picks=None,
#     filter_length="auto",
#     l_trans_bandwidth="auto",
#     h_trans_bandwidth="auto",
#     n_jobs=1,
#     method="fir",
#     iir_params=None,
#     copy=True,
#     phase="zero",
#     fir_window="hamming",
#     fir_design="firwin",
#     pad="reflect_limited",
#     verbose=None,
# ):
#
#     return filter.filter_data(x, sfreq, l_freq, h_freq)
#
# def len_split(len):
#
#     # TODO adapt to strange behavior of floating point 350 * 0.7 = 245 instead is giving 244.99999999999997
#
#     if len * 0.7 - int(len*0.7) == 0. and len * 0.15 - int(len*0.15) >= 0.:
#         if len * 0.15 - int(len*0.15) == 0.5:
#             train = round(len * 0.7)
#             valid = round(len * 0.15 + 0.1)
#             test = round(len * 0.15 - 0.1)
#         else:
#             train = round(len * 0.7)
#             valid = round(len * 0.15)
#             test = round(len * 0.15)
#
#     elif len * 0.7 - int(len*0.7) >= 0.5:
#         if len * 0.15 - int(len*0.15) >= 0.5:
#             train = round(len * 0.7)
#             valid = round(len * 0.15)
#             test = round(len * 0.15) - 1
#         else:
#             # round has a particular behavior on rounding 0.5
#             if len * 0.7 - int(len*0.7) == 0.5:
#                 train = round(len * 0.7 + 0.1)
#                 valid = round(len * 0.15)
#                 test = round(len * 0.15)
#             else:
#                 train = round(len * 0.7)
#                 valid = round(len * 0.15)
#                 test = round(len * 0.15)
#
#     else:
#         if len * 0.15 - int(len*0.15) >= 0.5:
#             train = round(len * 0.7)
#             valid = round(len * 0.15)
#             test = round(len * 0.15)
#         else:
#             train = round(len * 0.7)
#             valid = round(len * 0.15) + 1
#             test = round(len * 0.15)
#
#     return train, valid, test
#
