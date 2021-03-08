"""
    Utils used for import and process the MEG data.
    TODO: Integrate the bp approach and not in single functions.
"""

import errno
import os
import pickle
import sys

import h5py
import mne
import numpy as np
import torch
from mne.decoding import Scaler
from mne.decoding import UnsupervisedSpatialFilter
from numpy import trapz
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as skScaler


def bandpower_1d(data, sf, band, window_sec=None, relative=False):
    """
        Compute the average power of the signal x in a specific frequency band.
        https://raphaelvallat.com/bandpower.html
    Args:
        data (1d-array):
            Input signal in the time-domain.
        sf (float):
            Sampling frequency of the data.
        band (list):
            Lower and upper frequencies of the band of interest.
        window_sec (float):
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative (boolean):
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

    Returns:
        bp (float):
            Absolute or relative band power.
    """

    # band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


# def bandpower(x, fs, bands, window_sec=None, relative=True):
#         # x shape [n_epoch, n_ channle, n_times]
#         # bandpower [n_epoch, n_channel, 1]
#         n_epoch, n_channel, _ = x.shape
#         bp = np.zeros((n_epoch, n_channel, 1))
#         for idx, b in enumerate(bands):
#             print(b)
#             print(idx)
#             for epoch in range(n_epoch):
#                 for channel in range(n_channel):
#                     bp[epoch, channel] = bandpower_1d(x[epoch, channel, idx], fs, [fmax, fmin],
#                                                         window_sec=window_sec, relative=relative)
#
#         return bp


def bandpower(x, fs, fmin, fmax, window_sec=None, relative=True):
    """
    Compute the average power of the multi-channel signal x in a specific frequency band.
    Args:
        x (nd-array): [n_epoch, n_channel, n_times]
           The epoched input data.
        fs (float):
            Sampling frequency of the data.
        fmin (int): Low-band frequency.
        fmax (int): High-band frequency.
        window_sec (float):
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative (boolean):
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

    Returns:
        bp (nd-array): [n_epoch, n_channel, 1]
            Absolute or relative band power.
    """
    n_epoch, n_channel, _ = x.shape

    bp = np.zeros((n_epoch, n_channel, 1))
    for epoch in range(n_epoch):
        for channel in range(n_channel):
            bp[epoch, channel] = bandpower_1d(x[epoch, channel, :], fs, [fmin, fmax],
                                              window_sec=window_sec, relative=relative)

    return bp


def bandpower_multi(x, fs, bands, window_sec=None, relative=True):
    """
    Compute the average power of the multi-channel signal x in multiple frequency bands.
    Args:
        x (nd-array): [n_epoch, n_channel, n_times]
           The epoched input data.
        fs (float):
            Sampling frequency of the data.
        bands (list): list of bands to compute the bandpower. echa band is a tuple of fmin and fmax.
        window_sec (float):
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative (boolean):
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

    Returns:
        bp (nd-array): [n_epoch, n_channel, n_bands]
            Absolute or relative bands power.
    """
    n_epoch, n_channel, _ = x.shape
    bp_list = []
    for idx, band in enumerate(bands):
        fmin, fmax = band
        bp_list.append(bandpower(x, fs, fmin, fmax, window_sec=window_sec, relative=relative))

    bp = np.concatenate(bp_list, -1)

    return bp


def window_stack(x, window, overlap, sample_rate):
    """
    Epoched data manually. (suggested using MNE preimplemented functions)
    Args:
        x (nd-array): [n_channels, Times]
            Raw input data.
        window (float):
            Duration of the windows.
        overlap (float):
            Overlap value to generate overlapping windows.
        sample_rate (int):
            Sampling rate in ms.

    Returns:
        data (nd-array): [n_epochs, n_channels, n_times]
    """
    window_size = round(window * sample_rate)
    stride = round((window - overlap) * sample_rate)
    print(x.shape)
    print("window {}, stride {}, x.shape {}".format(window_size, stride, x.shape))

    return torch.cat([x[:, i: min(x.shape[1], i + window_size)] for i in range(0, x.shape[1], stride)], dim=1, )


def import_MEG(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement", rps=True):
    """
        Function that read the input files and epochs them using fix length overlapping windows. It returns the an
        array-like of the raw epoched data. It generates 1 event each (duration-overlap) s. Therefore, this value
        determine the prediction rate. It additionally retrives the y left and right target values as well as the
        bandpowers of the epoched input data. The input data are downsampled to a factor of 2.
    Args:
        raw_fnames [list]:
            List of path of files to import. (The main file format used is fif, however, it accept all the files
             accepted by mne.io.raw().
        duration (float):
            Length of the windows.
        overlap (float):
            Length of the overlap.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        y_measure (string):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        rps (bool):
            True, if generate bandpower spectrum.
            False, otherwise.

    Returns:
        X (nd-array): [n_epochs, n_channels, n_times]
            Output data.
        y_left (nd-array): [n_epochs, n_direction, n_times]
            Left hand target values. n_direction is normally 1 since the 2 original direction are combined with PCA.
            If not, n_direction is 2.
        y_right (nd-array): [n_epochs, , n_direction,  n_times]
            Right hand target values. n_direction is normally 1 since the 2 original direction are combined with PCA.
            If not, n_direction is 2.
        bp (nd-array): [n_epochs, n_channel, n_bands]
            Bandpowers values.
    """
    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            # Import raw data into mne.Raw
            raw = mne.io.Raw(fname, preload=True)
            # Generate fixed length events.
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
            # Isolate analysis to gradiometer and misc channels only
            raw.pick_types(meg='grad', misc=True)
            # Notch filter out some specific noisy bands
            raw.notch_filter([50, 100])
            # Band pass the input data
            raw.filter(l_freq=1., h_freq=70)
            # Get indices of accelerometer channels (y data)
            accelerometer_picks_left = mne.pick_channels(raw.info['ch_names'],
                                                         include=["MISC001", "MISC002"])
            accelerometer_picks_right = mne.pick_channels(raw.info['ch_names'],
                                                          include=["MISC003", "MISC004"])
            # Genrate eochs
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0), decim=2))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    # Concatenate all runs epochs in 1 structure
    epochs = mne.concatenate_epochs(epochs)
    # get indices of accelerometer channels

    # Pic only with gradiometer
    X = epochs.get_data()[:, :204, :]

    bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
    bp = bandpower_multi(X, fs=epochs.info['sfreq'], bands=bands, relative=True)

    # Normalize data
    if normalize_input:
        X = standard_scaling(X, scalings="mean", log=True)

    # Pick the y vales per each hand
    y_left = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_left, :]), measure=y_measure)
    y_right = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_right, :]), measure=y_measure)

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {}," \
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )
    return X, y_left, y_right, bp

def import_MEG_no_bp(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement"):
    """
        Function that read the input files and epochs them using fix length overlapping windows. It returns the an
        array-like of the raw epoched data. It generates 1 event each (duration-overlap) s. Therefore, this value
        determine the prediction rate. The input data are downsampled to a factor of 2.
    Args:
        raw_fnames [list]:
            List of path of files to import. (The main file format used is fif, however, it accept all the files
             accepted by mne.io.raw().
        duration (float):
            Length of the windows.
        overlap (float):
            Length of the overlap.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        y_measure (string):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        rps (bool):
            True, if generate bandpower spectrum.
            False, otherwise.

    Returns:
        X (nd-array): [n_epochs, n_channels, n_times]
            Output data.
        y_left (nd-array): [n_epochs, n_direction, n_times]
            Left hand target values. n_direction is normally 1 since the 2 original direction are combined with PCA.
            If not, n_direction is 2.
        y_right (nd-array): [n_epochs, , n_direction,  n_times]
            Right hand target values. n_direction is normally 1 since the 2 original direction are combined with PCA.
            If not, n_direction is 2.
    """
    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            # Import raw data into mne.Raw
            raw = mne.io.Raw(fname, preload=True)
            # Generate fixed length events.
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
            # Isolate analysis to gradiometer and misc channels only
            raw.pick_types(meg='grad', misc=True)
            # Notch filter out some specific noisy bands
            raw.notch_filter([50, 100])
            # Band pass the input data
            raw.filter(l_freq=1., h_freq=70)
            # Get indices of accelerometer channels (y data)
            accelerometer_picks_left = mne.pick_channels(raw.info['ch_names'],
                                                         include=["MISC001", "MISC002"])
            accelerometer_picks_right = mne.pick_channels(raw.info['ch_names'],
                                                          include=["MISC003", "MISC004"])
            # Genrate eochs
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0), decim=2))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    # Concatenate all runs epochs in 1 structure
    epochs = mne.concatenate_epochs(epochs)
    # get indices of accelerometer channels

    # Pic only with gradiometer
    X = epochs.get_data()[:, :204, :]

    # Normalize data
    if normalize_input:
        X = standard_scaling(X, scalings="mean", log=True)

    # Pick the y vales per each hand
    y_left = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_left, :]), measure=y_measure)
    y_right = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_right, :]), measure=y_measure)

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {}," \
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )
    return X, y_left, y_right



def import_MEG_Tensor(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement", rps=True):
    """
    Generate the epoched data as tensor to create the custom dataset for DL processing.
    TODO: Check X_out shape: may be [n_epochs,, 1, n_channels, n_times]
    Args:
        raw_fnames [list]:
            List of path of files to import. (The main file format used is fif, however, it accept all the files
             accepted by mne.io.raw().
        duration (float):
            Length of the windows.
        overlap (float):
            Length of the overlap.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        y_measure (str):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        rps (bool):
            True, if generate bandpower spectrum.
            False, otherwise.

    Returns:
        X (Tensor): [n_epochs, n_channels, n_times]
            Output data.
        y (Tensor): [n_epochs, n_direction*2, n_times]
            Left hand right target values stacked in 1 structures. n_direction is normally 1 since the 2 original
            direction are combined with PCA. If not, n_direction is 2.
        bp (Tensor): [n_epochs, n_channel, n_bands]
            Bandpowers values.
    """
    # Genrate the epoched data
    if rps:
        X, y_left, y_right, bp = import_MEG(raw_fnames, duration, overlap, normalize_input=normalize_input,
                                        y_measure=y_measure)
    else:
        X, y_left, y_right = import_MEG_no_bp(raw_fnames, duration, overlap, normalize_input=normalize_input,
                                            y_measure=y_measure)

    # Convert from Numpy nd-arrays to Pytorch Tensor.
    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
    y_left = torch.from_numpy(y_left.astype(np.float32))
    y_right = torch.from_numpy(y_right.astype(np.float32))

    if rps:
        bp = torch.from_numpy(bp.astype(np.float32))
        return X, torch.stack([y_left, y_right], dim=1), bp
    else:
        return X, torch.stack([y_left, y_right], dim=1)


def import_MEG_Tensor_form_file(data_dir, normalize_input=True, y_measure="movement"):
    """
    Import pre-epoched data. Data depends on previous values of duration and overlap.
    TODO: integrate bandpowers values.
    Args:
        data_dir (str):
            Directory where the epoched data structured is stored.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        y_measure (str:
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
    Returns:
        X (Tensor): [n_epochs, n_channels, n_times]
            Output data.
        y (Tensor): [n_epochs, n_direction*2, n_times]
            Left hand right target values stacked in 1 structures. n_direction is normally 1 since the 2 original
            direction are combined with PCA. If not, n_direction is 2.
    """
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
        "The input data are of shape: {}, the corresponding y_left shape is: {}," \
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )

    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

    y_left = torch.from_numpy(y_left.astype(np.float32))
    y_right = torch.from_numpy(y_right.astype(np.float32))

    return X, torch.stack([y_left, y_right], dim=1)


def filter_data(X, sampling_rate):
    """
    Implement filter of the input data. Suggested to use directly the mne.function on mne.Raw or mne.Epoch.
    Args:
         X (nd-array): [n_epochs, n_channels, n_times]
            Epoched data.
         sampling_rate (float):
            The sampling rate in Hz on which the data have been sampled.
    Returns:
        X_filtered (nd-array): [n_epochs, n_channels, n_times]
            Filtered data.
    """
    # TODO appropriate filtering and generalize function

    # careful with x shape, the last dimension should be n_times
    band_ranges = [(60, 200)]
    X_filtered = np.zeros((X.shape[0], X.shape[1] * len(band_ranges)), dtype=float)
    for index, band in enumerate(band_ranges):
        X_filtered[:, X.shape[1] * index: X.shape[1] * (index + 1)] = filter.filter_data(
            X, sampling_rate, band[0], band[1], method="fir"
        )

    return X_filtered


def split_data(X, y, test_size=0.3, random_state=0):
    """
    Split the epcohed data in train and test set.
    Args:
        X(nd-array): [n_epochs, n_channels, n_times]
            Input epoched data.
        y (nd-array) [n_epochs, n_direction*2, n_times]
            Y target values.
        test_size (int or float):
             If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples.
        random_state (int):
            Controls the shuffling applied to the data before applying the split.

    Returns:
        X_train (nd-array): [n_epochs*(1-test_size), n_channels, n_times]
        X_test (nd-array): [n_epochs*test_size, n_channels, n_times]
        y_train (nd-array): [n_epochs*(1-test_size), n_direction*2, n_times]
        y_test (nd-array): [n_epochs*test_size, n_direction*2, n_times]
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def save_skl_model(esitimator, models_path, name):
    """
    Save a sklearn model.
    Args:
        esitimator (sklearn.base.BaseEstimator):
            The model to save.
        models_path (str):
            Path to save the model at.
        name (str):
            File name of the model.

    """
    if os.path.exists(models_path):
        pickle.dump(esitimator, open(os.path.join(models_path, name), "wb"))
        print("Model saved successfully.")
    else:
        FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), models_path)


def load_skl_model(models_path):
    """
    Load a pikled model
    Args:
        models_path (str):
            Model path:

    Returns:
        model:
        The loaded model.
    """
    with open(models_path, "rb") as model:
        model = pickle.load(model)
        print("Model loaded successfully.")
        return model


def y_reshape(y, measure="movement", scaling=True):
    """
    Reshaping of the target such that there is one value per epoch to predict. The final approach is to generate a
    quantity of movement, summing all the instant acceleration values. (It is done on the n_times dimension)

    Args:
        y (nd-array): [n_epochs, 1, n_times] or [n_epochs, 2, n_times] or [n_epochs, n_direction*2, n_times]
        measure (str):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        scaling (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.

    Returns:
        y_reshaped: [n_epochs, ] or [n_epochs, 2] or [n_epochs, n_direction*2]
            Y values reshaped.

    """
    # Reshape using the mean of the n_times
    if measure == 'mean':
        y = np.sqrt(np.mean(np.power(y, 2), axis=-1))
        if scaling:
            y = standard_scaling(y, log=False)
    # Reshape summing across the n_times
    elif measure == 'movement':
        y = np.sum(np.abs(y), axis=-1)
        if scaling:
            y = standard_scaling(y, log=False)
    # Reshape integrating across the n_times
    elif measure == 'velocity':
        y = trapz(y, axis=-1) / y.shape[-1]
        if scaling:
            y = standard_scaling(y, log=False)
    # Reshape integrating twice across the n_times
    elif measure == 'position':
        vel = cumtrapz(y, axis=-1)
        y = trapz(vel, axis=-1) / y.shape[-1]
        if scaling:
            y = standard_scaling(y, log=False)

    else:
        raise ValueError("measure should be one of: mean, movement, velocity, position")

    return y.squeeze()


def y_PCA(y):
    """
    Apply PCA using the  mne.decoding.UnsupervisedSpatialFilter that applies PCA across time and samples. It comines the
    two accelerometers directions in 1 single hand-movement parameter.
    Args:
        y (nd-array): [n_epochs, 2, n_times]
            The 1 hand y values with two channels as directions-

    Returns:
         y_pca (nd-array): [n_epochs, 1, n_times]
    """
    pca = UnsupervisedSpatialFilter(PCA(1), average=False)

    return pca.fit_transform(y)


def y_reshape_final(y):
    """
    Function used to combine hand's direction dimensions.
    Args:
        y (nd-array): [n_epoch, n_channel, n_times]
            Epoched y values. For example, [n_epoch, 2, n_times]

    Returns:
        y (nd-array): [n_epoch, ]

    """
    # Combine the 2 directions using PCA
    pca = UnsupervisedSpatialFilter(PCA(1), average=False)
    y = pca.fit_transform(y)  # [n_epoch, 1, n_times]
    # sum abs values
    y = np.sum(np.abs(y), axis=-1) # [n_epoch, ]
    # standard-scale the values
    scaler = Scaler(scalings='mean')
    y = scaler.fit_transform(y)  # [n_epoch, ]

    return y.squeeze()


def save_pytorch_model(model, path, filename):
    """
    Save a Pytorch model.
    Args:
        model (nn.Module):
            The model to save.
        path (str):
            The path where to save the model at.
        filename:
            The model name.
    """
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
    """
    Load Pytorch model from state_dict().
    Args:
        model (nn.Module):
            The model to load the state_dict().
        path (str):
            The path of the model.
        device (str):
            The device used to train the model.

    Returns:
        model (nn.Module):
            The loaded model set in eval mode.
    """
    # model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(path))
    print("Model loaded from {}.".format(path))
    model.to(device)
    model.eval()
    return model


def normalize(data):
    """
    Normalize the data input between [-1, 1]
    Args:
        data (nd-array):
            The data to normalize.

    Returns:
        data (nd-array):
            The normalized data.

    """
    # linear rescale to range [0, 1]
    min = torch.min(data.view(data.shape[2], -1), dim=1, keepdim=True)[0]
    data -= min.view(1, 1, min.shape[0], 1)
    max = torch.max(data.view(data.shape[2], -1), dim=1, keepdim=True)[0]
    data /= max.view(1, 1, max.shape[0], 1)

    # Linear rescale to range [-1, 1]
    return 2 * data - 1


def standard_scaling(data, scalings="mean", log=False):
    """
    Standard scale the input data on last dimension. It center the data to 0 and scale to unit variance.
    It scales channel-wise estimating mu and sigma using all the epochs. Therefore it uses the mne.Scaler.
    Args:
        data (nd-array): [n_epochs, n_channel, n_times]
            The data to scale.
        scalings (str):
            If median it uses sklearn.preprocessing.RobustScaler.
            If mean' it uses sklearn.preprocessing.StandardScaler.
        log (bool):
            If True, apply log scaling before standardize.
            If False, do not apply log scaling.

    Returns:
        data (nd-array): [n_epochs, n_channel, n_times]
            The standardized data.
    """
    if log:
        data = np.log(data + np.finfo(np.float32).eps)

    if scalings in ["mean", "median"]:
        scaler = Scaler(scalings=scalings)
        data = scaler.fit_transform(data)
    else:
        raise ValueError("scalings should be mean or median")

    return data



def standard_scaling_sklearn(data):
    """
    Standard scale the input data on last dimension. It center the data to 0 and scale to unit variance.
    It scales trial- or epoch-wise, estimating the mean and the std using the timepoints of a single trial.
    Args:
        data (nd-array): [n_epochs, n_channel, n_times]
            The data to scale.

    Returns:
        data (nd-array): [n_epochs, n_channel, n_times]
            The standardized data.
    """
    n_epoch = data.shape[0]
    for e in range(n_epoch):
        scaler = skScaler()
        data[e, ...] = scaler.fit_transform(data[e, ...])

    return data

def transform_data():
    pass


def len_split(len):
    """
    Generate the splitting number of sample given the total dataset sample number.
    It split in train 70%, test and validation 15%.
    It take care of particular situation since the number of sample has to be passed perfectly to the random
    splitter fucntion such that test + train - valid == len
    Args:
        len (int):
        The dataset length.

    Returns:
        train (int):
            The train number of samples.
        valid (int):
            The validation number of samples.
        test (int):
            The test number of samples.

    TODO adapt to strange behavior of floating point 350 * 0.7 = 245 instead is giving 244.99999999999997
    TODO simplify the constraint as follow

    test = len * 0.7
    valid = len * 0.15
    tets = len - train - valid

    """

    train = int(len * 0.7)
    valid = int(len * 0.15)
    test = len - valid - train

    # if len * 0.7 - int(len * 0.7) == 0. and len * 0.15 - int(len * 0.15) >= 0.:
    #     if len * 0.15 - int(len * 0.15) == 0.5:
    #         train = round(len * 0.7)
    #         valid = round(len * 0.15 + 0.1)
    #         test = round(len * 0.15 - 0.1)
    #     else:
    #         train = round(len * 0.7)
    #         valid = round(len * 0.15)
    #         test = round(len * 0.15)
    #
    # elif len * 0.7 - int(len * 0.7) >= 0.5:
    #     if len * 0.15 - int(len * 0.15) >= 0.5:
    #         train = round(len * 0.7)
    #         valid = round(len * 0.15)
    #         test = round(len * 0.15) - 1
    #     else:
    #         # round has a particular behavior on rounding 0.5
    #         if len * 0.7 - int(len * 0.7) == 0.5:
    #             train = round(len * 0.7 + 0.1)
    #             valid = round(len * 0.15)
    #             test = round(len * 0.15)
    #         else:
    #             train = round(len * 0.7)
    #             valid = round(len * 0.15)
    #             test = round(len * 0.15)
    #
    # else:
    #     if len * 0.15 - int(len * 0.15) >= 0.5:
    #         train = round(len * 0.7)
    #         valid = round(len * 0.15)
    #         test = round(len * 0.15)
    #     else:
    #         train = round(len * 0.7)
    #         valid = round(len * 0.15) + 1
    #         test = round(len * 0.15)

    return train, valid, test

def import_MEG_2(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement"):
    """
            Function that read the input files and epochs them using fix length overlapping windows. It returns the an
            array-like of the raw epoched data. It generates 1 event each (duration-overlap) s. Therefore, this value
            determine the prediction rate. The input data are downsampled to a factor of 2. Version wit 2 directions
            as target.

        Args:
            raw_fnames [list]:
                List of path of files to import. (The main file format used is fif, however, it accept all the files
                 accepted by mne.io.raw().
            duration (float):
                Length of the windows.
            overlap (float):
                Length of the overlap.
            normalize_input (bool):
                True, if apply channel-wise standard scaling to the input data.
                False, does not apply ay normalization.
            y_measure (string):
                Measure used to reshape the y. Values in [mean, movement, velocity, position]
            rps (bool):
                True, if generate bandpower spectrum.
                False, otherwise.

        Returns:
            X (nd-array): [n_epochs, n_channels, n_times]
                Output data.
            y_left (nd-array): [n_epochs, n_direction, n_times]
                Left hand target values. n_direction is normally 1 since the 2 original direction are combined with PCA.
                If not, n_direction is 2.
            y_right (nd-array): [n_epochs, , n_direction,  n_times]
                Right hand target values. n_direction is normally 1 since the 2 original direction are combined with PCA.
                If not, n_direction is 2.
        """
    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            raw = mne.io.Raw(fname, preload=True)
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
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0), decim=2))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)

    epochs = mne.concatenate_epochs(epochs)
    # get indices of accelerometer channels

    # pic only with gradiometer
    X = epochs.get_data()[:, :204, :]

    bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
    bp = bandpower_multi(X, fs=epochs.info['sfreq'], bands=bands, relative=True)

    if normalize_input:
        X = standard_scaling(X, scalings="mean", log=True)

    y_left = y_reshape(epochs.get_data()[:, accelerometer_picks_left, :], measure=y_measure)
    y_right = y_reshape(epochs.get_data()[:, accelerometer_picks_right, :], measure=y_measure)

    # y_left = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_left, :]), measure=y_measure)
    # y_right = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_right, :]), measure=y_measure)

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {},"\
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )
    return X, y_left, y_right, bp


def import_MEG_Tensor_2(raw_fnames, duration, overlap, normalize_input=True, y_measure="movement"):
    """
        Generate the epoched data as tensor to create the custom dataset for DL processing with the 2 directions as
        target.

        Args:
            raw_fnames [list]:
                List of path of files to import. (The main file format used is fif, however, it accept all the files
                 accepted by mne.io.raw().
            duration (float):
                Length of the windows.
            overlap (float):
                Length of the overlap.
            normalize_input (bool):
                True, if apply channel-wise standard scaling to the input data.
                False, does not apply ay normalization.
            y_measure (str):
                Measure used to reshape the y. Values in [mean, movement, velocity, position]
            rps (bool):
                True, if generate bandpower spectrum.
                False, otherwise.

        Returns:
            X (Tensor): [n_epochs, n_channels, n_times]
                Output data.
            y (Tensor): [n_epochs, n_direction*2, n_times]
                Left hand right target values stacked in 1 structures. n_direction is normally 1 since the 2 original
                direction are combined with PCA. If not, n_direction is 2.
            bp (Tensor): [n_epochs, n_channel, n_bands]
                Bandpowers values.
        """

    X, y_left, y_right, bp = import_MEG_2(raw_fnames, duration, overlap, normalize_input=normalize_input, y_measure=y_measure)

    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

    y_left = torch.from_numpy(y_left.astype(np.float32))
    y_right = torch.from_numpy(y_right.astype(np.float32))

    bp = torch.from_numpy(bp.astype(np.float32))
    return X, torch.stack([y_left, y_right], dim=1), bp


def import_MEG_cross_subject_train(data_dir, file_name, subject, hand=0, y="pca"):
    """
    Import the data and generate the train set.
    Test set composed by input subject.
    Train set composed by all the others.
    Args:
        data_dir (string):
            Path of the data directory.
        file_name (string):
            Data file name. file.hdf5.
        subject (int):
            Number of the test subject.
        hand (int):
            Which hand to use during. 0 = left, 1 = right.
        y (string):
            The target variable. The value can be pca, left_single.
            Left_pca: pca to combine the 2 direction of the left hand. Standard scaled channel-wised. Abs-sum to epoch.

    Returns:
        X_train, y_train, rps_train
    """
    if y not in ["pca", "left_single_1"]:
        raise ValueError("The y value to predict does not exist.")

    if y == "pca" and hand == 0:
        y = "left_pca"
        print("y measure : ", y)

    if y == "pca" and hand == 1:
        y = "right_pca"
        print("y measure : ", y)

    X_train = []
    rps_train = []
    y_train = []

    with h5py.File("".join([data_dir, file_name]), "r") as f:
        subjects = f.keys()
        if y == "left_pca":
            for sub in subjects:
                if sub != ("sub" + str(subject)) and sub != "sub4":
                    X_train.append(f[sub]["MEG"][...])
                    rps_train.append(f[sub]["RPS"][...])
                    y_train.append(f[sub]["Y_left"][...])

        if y == "left_single_1":
            for sub in subjects:
                if sub != ("sub" + str(subject)) and sub != "sub4":
                    X_train.append(f[sub]["MEG"][...])
                    rps_train.append(f[sub]["RPS"][...])
                    y_train.append(y_reshape(np.expand_dims(f[sub]["ACC_original"][:, 0, :], 1), scaling=True))

        if y == "right_pca":
            for sub in subjects:
                if sub != ("sub" + str(subject)) and sub != "sub3" and sub != "sub5":
                    X_train.append(f[sub]["MEG"][...])
                    rps_train.append(f[sub]["RPS"][...])
                    y_train.append(f[sub]["Y_right"][...])

    X_train = torch.from_numpy(np.concatenate(X_train))
    rps_train = torch.from_numpy(np.concatenate(rps_train))
    y_train = torch.from_numpy(np.concatenate(y_train))

    print(X_train.shape)
    print(y_train.shape)


    return X_train.unsqueeze(1), y_train.unsqueeze(-1).repeat(1, 2), rps_train


def import_MEG_cross_subject_test(data_dir, file_name, subject, hand = 0,  y="pca"):
    """
    Import the data and generate the test set.
    Test set composed by input subject.
    Train set composed by all the others.
    Args:
        data_dir (string):
            Path of the data directory.
        file_name (string):
            Data file name. file.hdf5.
        subject (int):
            Number of the test subject.
        hand (int):
            Which hand to use during. 0 = left, 1 = right.
        y (string):
            The target variable. The value can be left_pca, left_single.
            Left_pca: pca to combine the 2 direction of the left hand. Standard scaled channel-wised. Abs-sum to epoch.

    Returns:
        X_test, y_test, rps_test
    """

    if y not in ["pca", "left_single_1"]:
        raise ValueError("The y value to predict does not exist.")

    if y == "pca" and hand == 0:
        y = "left_pca"

    if y == "pca" and hand == 1:
        y = "right_pca"

    print("Test subject: sub" + str(subject) + "!")
    sub = "sub" + str(subject)

    with h5py.File("".join([data_dir, file_name]), "r") as f:
        if y == "left_pca":
            X_test = f[sub]["MEG"][...]
            rps_test = f[sub]["RPS"][...]
            y_test = f[sub]["Y_left"][...]

        if y == "left_single_1":
            X_test = f[sub]["MEG"][...]
            rps_test = f[sub]["RPS"][...]
            y_test = y_reshape(np.expand_dims(f[sub]["ACC_original"][:, 0, :], 1), scaling=True)

        if y == "right_pca":
            X_test = f[sub]["MEG"][...]
            rps_test = f[sub]["RPS"][...]
            y_test = f[sub]["Y_right"][...]



    X_test = torch.from_numpy(X_test)
    rps_test = torch.from_numpy(rps_test)
    y_test = torch.from_numpy(y_test)

    return X_test.unsqueeze(1), y_test.unsqueeze(-1).repeat(1, 2), rps_test

def len_split_cross(len):
    """
    Generate the splitting number of sample given the total dataset sample number.
    It split in train 80%, test and validation 20%.
    Args:
        len (int):
        The dataset length.

    Returns:
        train (int):
            The train number of samples.
        valid (int):
            The validation number of samples.

    """

    train = int(len * 0.80)
    valid = len - train

    return train, valid


def import_MEG_within_subject(data_dir, file_name, subject, hand=0,  y="pca"):
    """
    Import the data and generate the X, y, and bp tensors.
    Args:
        data_dir (string):
            Path of the data directory.
        file_name (string):
            Data file name. file.hdf5.
        subject (int):
            Number of the test subject.
        hand (int):
            Which hand to use during. 0 = left, 1 = right.
        y (string):
            The target variable. The value can be left_pca, left_single.
            Left_pca: pca to combine the 2 direction of the left hand. Standard scaled channel-wised. Abs-sum to epoch.

    Returns:
        X_test, y_test, rps_test
    """

    if y not in ["pca", "left_single_1"]:
        raise ValueError("The y value to predict does not exist.")

    if y == "pca" and hand == 0:
        y = "left_pca"

    if y == "pca" and hand == 1:
        y = "right_pca"

    sub = "sub" + str(subject)
    print(sub)

    with h5py.File("".join([data_dir, file_name]), "r") as f:
        if y == "left_pca":
            X = f[sub]["MEG"][...]
            rps = f[sub]["RPS"][...]
            y = f[sub]["Y_left"][...]

        if y == "left_single_1":
            X = f[sub]["MEG"][...]
            rps = f[sub]["RPS"][...]
            y = y_reshape(np.expand_dims(f[sub]["ACC_original"][:, 0, :], 1), scaling=True)

        if y == "right_pca":
            X = f[sub]["MEG"][...]
            rps = f[sub]["RPS"][...]
            y = f[sub]["Y_right"][...]



    X = torch.from_numpy(X)
    rps = torch.from_numpy(rps)
    y = torch.from_numpy(y)

    return X.unsqueeze(1), y.unsqueeze(-1).repeat(1, 2), rps


def import_MEG_within_subject_ivan(data_path, subject=8, hand=0, mode="train"):
    """
    Import the data and generate the X, y, and bp tensors.
    Args:
        data_dir (string):
            Path of the data directory.
        file_name (string):
            Data file name. file.hdf5.
        subject (int):
            Number of the test subject.
        hand (int):
            Which hand to use during. 0 = left, 1 = right.
        y (string):
            The target variable. The value can be left_pca, left_single.
            Left_pca: pca to combine the 2 direction of the left hand. Standard scaled channel-wised. Abs-sum to epoch.

    Returns:
        X_test, y_test, rps_test
    """
    file_name = "ball_left_mean.npz"
    rps_name = "rps.npz"
    print("loading dataset for {} ".format(mode))

    dataset = np.load(os.path.join(data_path, file_name))
    rps_data = np.load(os.path.join(data_path, rps_name))
    print("datasets :", dataset.files)
    print("rps :", rps_data.files)

    if mode == "train":
        X = dataset["X_train"]
        y = dataset["y_train"]
        rps = rps_data["rps_train"]


    elif mode == "val":
        X = dataset["X_val"]
        y = dataset["y_val"]
        rps = rps_data["rps_val"]

    elif mode == "test":
        X = dataset["X_test"]
        y = dataset["y_test"]
        rps = rps_data["rps_test"]

    else:
        raise ValueError("mode value must be train, val or test!")


    X = np.swapaxes(X, 2, -1) # To reshape the data [n_epoch, 1, n_channel, n_times]
    print(X.shape)

    # bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
    # rps = bandpower_multi(X.squeeze(), fs=250, bands=bands, relative=True)

    # local
    # rps = np.ones([X.shape[0], 204, 6])

    # generate rps

    X = torch.from_numpy(X).float()
    rps = torch.from_numpy(rps).float()
    y = torch.from_numpy(y).float()

    return X, y.repeat(1, 2), rps