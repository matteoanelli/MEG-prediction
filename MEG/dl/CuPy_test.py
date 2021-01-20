#!/usr/bin/env python
"""
    Script to test CuPy to speed up rps integration.
"""
import os
import argparse


import numpy as np
import mne
import mneflow
import time as timer

from numpy import trapz
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from scipy.signal import welch


def segment_x(data, segment_length=200, stride=None):
    """Split the data into fixed-length segments.

    Parameters
    ----------
    data : ndarray
        Data array of shape (n_epochs, n_channels, n_times)

    segment_length : int or False
        Length of segment into which to split the data in time samples.

    stride: int or None
        Stride value to slide the window in time points. (stride = length - overlap)

    Returns
    -------
    data : list of ndarrays [n_epoch_fix_lenght, n_channels, segment_length]
        TODO: fix formula
        where n_epoch_fix_lenght = (n_epochs//seq_length)*(n_times - segment_length + 1)//stride
        """
    x_out = []
    # If not overlapping window
    if not stride:
            stride = segment_length

    # print('data :',data.shape)

    for jj, xx in enumerate(data):
        # print('xx :', xx.shape)
        n_ch, n_t = xx.shape
        last_segment_start = n_t - segment_length
        #print('last start:', last_segment_start)

        #print("stride:", stride)
        starts = np.arange(0, last_segment_start+1, stride)
        ## TODO: check if the segment_lenght is right
        segments = [xx[..., s:s+segment_length] for s in starts]

        x_new = np.stack(segments, axis=0)
        # print('x_new shape : ', x_new.shape)

            #print("x_new:", x_new.shape)
#        if jj == len(data) - 1:
#            print("n_segm:", seq_length)
#            print("x_new:", x_new.shape)
        x_out.append(x_new)

    #print(len(x_out))
    if len(x_out) > 1:
        X = np.concatenate(x_out)
    else:
        X = x_out[0]
    print("X:", X.shape)
    return X


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


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir
    sub = 8

    # Generate the data input path list. Each subject has 3 runs stored in 3 different files.
    subj_id = "/sub"+str(sub)+"/ball0"
    raw_fnames = [["".join([data_dir, subj_id, str(i), "_sss_trans.fif"]) for i in range(1 if sub != 3 else 2, 4)][0]]
    print(raw_fnames)

    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            # Import raw data into mne.Raw
            raw = mne.io.Raw(fname, preload=True)
            # Generate fixed length events.
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            # events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
            events = mne.pick_events(mne.find_events(raw), include=[3, 5])
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
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=20., baseline=(0, 0), decim=2))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    # Concatenate all runs epochs in 1 structure
    epochs = mne.concatenate_epochs(epochs)

    
    X = segment_x(epochs.get_data()[:, :204, :], 500, stride=100)

    print("X shape: ", X.shape)

    start_time = timer.time()

    bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
    bp = bandpower_multi(X, fs=epochs.info['sfreq'], bands=bands, relative=True)

    total_time = timer.time() - start_time
    print("RPS done in {:.4f}".format(total_time))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# Directories

	parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
    	help="Input data directory (default= Z:\Desktop\\)")
	parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
    	help="Figure data directory (default= MEG\Figures)")
	parser.add_argument('--model_dir', type=str, default='MEG\Models',
    	help="Model data directory (default= MEG\Models\)")


	args = parser.parse_args()

	main(args)