#!/usr/bin/env python

"""
    Plot bp only sub 8

"""

import argparse
import os
import sys

import mne
import h5py
import time as timer
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, r"")

from MEG.Utils.utils import (
    bandpower_multi,
    standard_scaling_sklearn,
    y_PCA,
    y_reshape,
)

if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument("--sub", type=int, default="8",
                        help="Sub number (default= 8)",)
    parser.add_argument("--hand", type=int, default="0",
                        help="Hand (default= 0)", )

    # Directories
    parser.add_argument("--data_dir", type=str, default="Z:\Desktop\\",
                        help="Input data directory (default= Z:\Desktop\\)",)
    parser.add_argument("--figure_dir", type=str, default="Z:\Desktop\\",
                        help="Input data directory (default= Z:\Desktop\\)",)

    # Duration and overlap
    parser.add_argument("--duration", type=float, default=1.0, metavar="N",
                        help="Duration of the time window  (default: 1s)",)
    parser.add_argument("--overlap", type=float, default=0.8, metavar="N",
                        help="overlap of time window (default: 0.8s)",)

    args = parser.parse_args()

    data_dir = args.data_dir
    figure_dir = args.figure_dir

    hand = args.hand
    subject = str(args.sub)

    local = False

    start = timer.time()
    print("Pre-processing data....")
    # Generate the data input path list. Each subject has 3 runs stored in 3
    # different files.
    if local:
        # local
        subj_id = "/sub" + str(args.sub) + "/ball"
        raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"])
                  for i in range(1, 2)]

        epochs = []
        for fname in raw_fnames:
            if os.path.exists(fname):
                raw = mne.io.Raw(fname, preload=True).crop(tmax=30)
                # raw = mne.io.Raw(fname, preload=True)
                # events = mne.find_events(raw, stim_channel='STI101',
                # min_duration=0.003)
                events = mne.make_fixed_length_events(
                    raw, duration=args.duration, overlap=args.overlap)
                raw.pick_types(meg="grad", misc=True)
                raw.notch_filter([50, 100])
                raw.filter(l_freq=1.0, h_freq=70)

                # get indices of accelerometer channels
                epochs.append(mne.Epochs(raw, events, tmin=0.0,
                                         tmax=args.duration, baseline=(0, 0),
                                         decim=2,))
                del raw
            else:
                print("No such file '{}'".format(fname), file=sys.stderr)
        epochs = mne.concatenate_epochs(epochs)

        X = epochs.get_data()[:40, :20, :]

        X = standard_scaling_sklearn(X)

        print("X shape before saving: {}".format(X.shape))

        # Keep only the first 6 misc channel (accelerometer data)
        accelermoters = epochs.get_data()[:, 204:210, :]

    # RPS
        bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
        bp = bandpower_multi(X, fs=epochs.info["sfreq"], bands=bands,
                             relative=True)

    else:
        # analysing the x data.
        if hand == 0:
            file_name = "sub_{}_left.npz".format(str(subject))
            print("processing file :", file_name)
            out_file = "sub_{}_left_rps.npz".format(str(subject))
            print("output_file: ", out_file)
        else:
            file_name = "sub_{}_right.npz".format(str(subject))
            print("processing file :", file_name)
            out_file = "sub_{}_right_rps.npz".format(str(subject))
            print("output_file: ", out_file)

        dataset = np.load(os.path.join(data_dir, file_name))

        X = dataset["X_train"]
        X = np.swapaxes(X, 2, -1)

        print("training dataset of shape:", X.shape)
        print("############################################################")
        print("global mean", np.mean(X))
        print("global min", np.min(X))
        print("global max", np.max(X))
        print("global std", np.std(X))
        print("############################################################")
        print("mean per first 100 epoch", np.mean(X[:100, ...], axis=(1, 2)))
        print("min per first 100 epoch", np.min(X[:100, ...], axis=(1, 2)))
        print("max per first 100 epoch", np.max(X[:100, ...], axis=(1, 2)))
        print("mean per 2 epochs all channels", np.mean(X[99:101, ...], axis=(2)))
        print("min per first 100 epoch", np.min(X[99:101, ...], axis=(2)))
        print("############################################################")
        print("max per first 100 epoch", np.max(X[99:101, ...], axis=(2)))
        print("X shape before saving: {}".format(X.shape))
        print("############################################################")
        print("print 2 random epoch for 2 channels", X[99:100, 100:102, :])
        print("############################################################")
        print("print 2 random epoch for 2 channels", X[200:202, 80:82, :])


        if hand == 0:
            rps_name = "sub_{}_left_rps.npz".format(str(subject))
        else:
            rps_name = "sub_{}_left_rps.npz".format(str(subject))

        rps_data = np.load(os.path.join(data_dir, rps_name))
        print("rps :", rps_data.files)

        bp = rps_data["rps_train"]
        print("bp_statistics")
        print("global mean", np.mean(bp))
        print("global min", np.min(bp))
        print("global max", np.max(bp))
        print("global std", np.std(bp))

    print("bp shape :", bp.shape)
    print(bp[30, ...].shape)
    for epoch in [1, 30, 90, 100]:

        fig = plt.figure(figsize=[10, 4])
        im = plt.pcolormesh(bp[epoch, ...])
        fig.colorbar(im)
        plt.title("RPS_epoch_{}".format(epoch))
        plt.ylabel("Channels")
        plt.xlabel("Bands")
        plt.locator_params(axis="y", nbins=5)
        plt.xticks(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            ["delta", "tetha", "low-alpha", "high-alpha", "beta", "low-gamma"],
        )
        plt.savefig(os.path.join(figure_dir, "RPS_epoch_{}_hand_{}.pdf"
            .format(epoch, "right" if hand == 1 else "left")))
        # plt.show()

    # y reshape in one pca direction. After pca the two directions it firstly reshape them and
    # eventually standard scale them.
    # y_left = y_reshape(y_PCA(epochs.get_data()[:, 204:206, :]))
