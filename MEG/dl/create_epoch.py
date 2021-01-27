#!/usr/bin/env python

"""
    Script to save the epoched data.

"""

import argparse
import os
import sys

import mne
import h5py
import time as timer
import numpy as np

sys.path.insert(1, r'')

from MEG.Utils.utils import bandpower_multi, standard_scaling_sklearn, y_PCA, y_reshape

if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--out_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")

    # Duration and overlap
    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.8, metavar='N',
                        help='overlap of time window (default: 0.8s)')

    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    start = timer.time()
    print("Pre-processing data....")
    # Generate the data input path list. Each subject has 3 runs stored in 3 different files.
    subj_id = "/sub" + str(args.sub) + "/ball0"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss_trans.fif"]) for i in range(1 if args.sub != 3 else 2, 4)]

    # local
    # subj_id = "/sub"+str(args.sub)+"/ball"
    # raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]

    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            raw = mne.io.Raw(fname, preload=True).crop(tmax=60)
            # raw = mne.io.Raw(raw_fnames[0], preload=True)
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            events = mne.make_fixed_length_events(raw, duration=args.duration, overlap=args.overlap)
            raw.pick_types(meg='grad', misc=True)
            raw.notch_filter([50, 100])
            raw.filter(l_freq=1., h_freq=70)

            # get indices of accelerometer channels
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=args.duration, baseline=(0, 0),  decim=2))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    epochs = mne.concatenate_epochs(epochs)

    X = epochs.get_data()[:, :204, :]

    X = standard_scaling_sklearn(X)

    print('X shape before saving: {}'.format(X.shape))

    # Keep only the first 6 misc channel (accelerometer data)
    accelermoters = epochs.get_data()[:, 204:210, :]

    # RPS
    bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
    bp = bandpower_multi(X, fs=epochs.info['sfreq'], bands=bands, relative=True)

    # y reshape in one pca direction. After pca the two directions it firstly reshape them and
    # eventually standard scale them.
    y_left = y_reshape(y_PCA(epochs.get_data()[:, 204:206, :]))

    print("Pre-processing done in: {}".format(timer.time()-start))

    print("Begin to save the dataset on disk")
    start = timer.time()

    if os.path.exists("".join([out_dir,"\data.hdf5"])):
        with h5py.File("".join([out_dir,"\data.hdf5"]), "a") as f:
            grp1 = f.create_group("".join(["sub" + str(args.sub)]))
            grp1.create_dataset("MEG", data=X, dtype='f')
            grp1.create_dataset("ACC_original", data=accelermoters, dtype='f')
            grp1.create_dataset("Y_left", data=y_left, dtype='f')
            grp1.create_dataset("RPS", data=bp, dtype='f')
    else:
        with h5py.File("".join([out_dir,"\data.hdf5"]), "w") as f:
            grp1 = f.create_group("".join(["sub" + str(args.sub)]))
            grp1.create_dataset("MEG", data=X, dtype='f')
            grp1.create_dataset("ACC_original", data=accelermoters, dtype='f')
            grp1.create_dataset("Y_left", data=y_left, dtype='f')
            grp1.create_dataset("RPS", data=bp, dtype='f')

    print("Data saved in: {}".format(timer.time()-start))

    with h5py.File("".join([out_dir, "\data.hdf5"]), "r") as f:
        print(f)
        print(f.keys())
        for group in f.keys():
            print("/{}".format(group))
            for dset in f[group].keys():
                print("{}/{}/{}".format(f.name, group, dset))

