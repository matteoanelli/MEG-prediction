#!/usr/bin/env python

import os, sys
import argparse
import numpy as np
from mne.time_frequency import psd_array_welch
sys.path.insert(1, r"")

from MEG.Utils.utils import *

def generate_welch(data, fmin=1, fmax=70, fs=250, n_jobs=1):

    psds, freqs = psd_array_welch(data, fs, fmin, fmax, n_per_seg=int(fs/2),
                          n_overlap=int(fs/4), n_jobs=n_jobs)

    return np.expand_dims(psds, 1)

def main(args):

    data_dir = args.data_dir

    sub = args.sub
    hand = args.hand


    if hand == 0:
        file_name = "sub_{}_left.npz".format(str(sub))
        print("processing file :", file_name)
        out_file = "sub_{}_left_welch.npz".format(str(sub))
        print("output_file: ", out_file)
    else:
        file_name = "sub_{}_right.npz".format(str(sub))
        print("processing file :", file_name)
        out_file = "sub_{}_right_welch.npz".format(str(sub))
        print("output_file: ", out_file)

    dataset = np.load(os.path.join(data_dir, file_name))

    X_train = dataset["X_train"]
    X_train = np.swapaxes(X_train, 2, -1)
    X_val = dataset["X_val"]
    X_val = np.swapaxes(X_val, 2, -1)
    X_test = dataset["X_test"]
    X_test = np.swapaxes(X_test, 2, -1)



    welch_train = generate_welch(X_train.squeeze(), fs=250, fmin=1,
                                 fmax=70, n_jobs=1)
    print("welch train data shape: ", welch_train.shape)
    print("train_done")

    welch_val = generate_welch(X_val.squeeze(), fs=250, fmin=1,
                                 fmax=70, n_jobs=1)
    print("welch valid data shape: ", welch_val.shape)
    print("valid_done")

    welch_test = generate_welch(X_test.squeeze(), fs=250, fmin=1,
                                 fmax=70, n_jobs=1)
    print("welch test data shape: ", welch_test.shape)
    print("test_done")

    np.savez(os.path.join(data_dir, out_file), welch_train=welch_train,
             welch_val=welch_val, welch_test=welch_test)



if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")
    parser.add_argument('--hand', type=int, default='0',
                        help="hand (default= 0)")


    args = parser.parse_args()

    main(args)
