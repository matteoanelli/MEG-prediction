#!/usr/bin/env python

"""
    Script to save the epoched data.

    TODO: Better implementation and testing.
"""

import argparse
import os
import sys

import mne
import numpy as np

sys.path.insert(1, r"")

if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument("--sub", type=int, default="8", help="Input data directory (default= 8)")

    # Directories
    parser.add_argument(
        "--data_dir", type=str, default="Z:\Desktop\\", help="Input data directory (default= Z:\Desktop\\)"
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, metavar="N", help="Duration of the time window  (default: 1s)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.8, metavar="N", help="overlap of time window (default: 0.8s)"
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    subj_id = "/sub" + str(args.sub) + "/ball"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 4)]

    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            raw = mne.io.Raw(raw_fnames[0], preload=True)
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            events = mne.make_fixed_length_events(raw, duration=args.duration, overlap=args.overlap)
            raw.pick_types(meg="grad", misc=True)
            raw.notch_filter([50, 100])
            raw.filter(l_freq=1.0, h_freq=70)

            # get indices of accelerometer channels
            accelerometer_picks_left = mne.pick_channels(raw.info["ch_names"], include=["MISC001", "MISC002"])
            accelerometer_picks_right = mne.pick_channels(raw.info["ch_names"], include=["MISC003", "MISC004"])
            epochs.append(mne.Epochs(raw, events, tmin=0.0, tmax=args.duration, baseline=(0, 0)))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    epochs = mne.concatenate_epochs(epochs)

    X = epochs.get_data()[:, :204, :]

    print("X shape before saving: {}".format(X.shape))

    X.tofile("".join([data_dir, "sub" + str(args.sub), "\X.dat"]))

    y_left = epochs.get_data()[:, accelerometer_picks_left, :]
    y_right = epochs.get_data()[:, accelerometer_picks_right, :]

    print("y_left shape before saving: {}".format(y_left.shape))
    print("y_right shape before saving: {}".format(y_right.shape))

    y_left.tofile("".join([data_dir, "sub" + str(args.sub), "\y_left.dat"]))
    y_right.tofile("".join([data_dir, "sub" + str(args.sub), "\y_right.dat"]))

    X = np.fromfile("".join([data_dir, "sub" + str(args.sub), "\X.dat"]), dtype=float)
    print("X shape after saving: {}".format(X.shape))

    y_left = np.fromfile("".join([data_dir, "sub" + str(args.sub), "\y_left.dat"]), dtype=float)
    y_right = np.fromfile("".join([data_dir, "sub" + str(args.sub), "\y_right.dat"]), dtype=float)

    print("y_left shape after saving: {}".format(y_left.shape))
    print("y_right shape after saving: {}".format(y_right.shape))
