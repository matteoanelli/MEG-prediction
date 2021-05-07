#!/usr/bin/env python
"""
    Script to add dataset to each subject n the data.hdf5 file.

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
from sklearn.preprocessing import StandardScaler
from MEG.Utils.utils import standard_scaling_sklearn, y_reshape, y_PCA

if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Z:\Desktop\\",
        help="Input data directory (default= Z:\Desktop\\)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Z:\Desktop\\",
        help="Input data directory (default= Z:\Desktop\\)",
    )

    args = parser.parse_args()

    duration = 1.0
    overlap = 0.8

    data_dir = args.data_dir
    out_dir = args.out_dir
    file_name = "data.hd5f"

    # subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # local
    subjects = [1, 5, 8]

    for sub in subjects:

        print("Processing sub:", sub)

        subj_id = "/sub" + str(sub) + "/ball0"
        raw_fnames = [
            "".join([data_dir, subj_id, str(i), "_sss_trans.fif"])
            for i in range(1 if sub != 3 else 2, 4)
        ]

        # local
        subj_id = "/sub" + str(sub) + "/ball"
        raw_fnames = [
            "".join([data_dir, subj_id, str(i), "_sss.fif"])
            for i in range(1, 2)
        ]

        epochs = []
        for fname in raw_fnames:
            if os.path.exists(fname):
                # raw = mne.io.Raw(fname, preload=True).crop(tmax=60)
                raw = mne.io.Raw(fname, preload=True)
                # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
                events = mne.make_fixed_length_events(
                    raw, duration=duration, overlap=overlap
                )
                raw.pick_types(meg="grad", misc=True)
                raw.notch_filter([50, 100])
                raw.filter(l_freq=1.0, h_freq=70)

                # get indices of accelerometer channels
                epochs.append(
                    mne.Epochs(
                        raw,
                        events,
                        tmin=0.0,
                        tmax=duration,
                        baseline=(0, 0),
                        decim=2,
                    )
                )
                del raw
            else:
                print("No such file '{}'".format(fname), file=sys.stderr)
        epochs = mne.concatenate_epochs(epochs)

        # Keep only the first 6 misc channel (accelerometer data)
        accelermoters = epochs.get_data()[:, 204:210, :]

        # right hand

        if sub in [1, 2, 4]:
            # y reshape in one pca direction. After pca the two directions it firstly reshape them and
            # eventually standard scale them.
            y_right = y_reshape(y_PCA(epochs.get_data()[:, 207:209, :]))

        elif sub in [6, 7, 8, 9]:
            y_right = y_reshape(y_PCA(epochs.get_data()[:, 206:208, :]))

        else:
            # sub not properly recorded
            y_right = np.zeros((accelermoters.shape[0],))

        with h5py.File("".join([out_dir, "data.hdf5"]), "a") as f:
            grp1 = f["".join(["sub" + str(sub)])]
            grp1.create_dataset("Y_right", data=y_right, dtype="f")
            # data = f["sub" + str(sub) + "/Y_right"]
            # data = y_right

    with h5py.File("".join([out_dir, "data.hdf5"]), "r") as f:
        print(f)
        print(f.keys())
        for group in f.keys():
            print("/{}".format(group))
            for dset in f[group].keys():
                print("{}/{}/{}".format(f.name, group, dset))

    # process the new y (y_trial)

    # add right_hand pca channel
