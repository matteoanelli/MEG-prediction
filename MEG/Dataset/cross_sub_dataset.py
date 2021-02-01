#!/usr/bin/env python

"""
    Script to create the different train and test sets.
    Implementing leave-one-out subject.
    Test set: 1 subject.
    Trin set: 80% other subjects
    Valid set: 20% other subjects

    # TODO: not compleated...    UNCOMPLETE

"""

import argparse
import sys

import h5py
from sklearn.model_selection import train_test_split

sys.path.insert(1, r'')

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

    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    with h5py.File("".join([out_dir, "\data.hdf5"]), "r") as f:
        groups = f.keys()
        for group in groups:
            X = f[group]["MEG"]

            rps = f[group]["RPS"]

            Y_left = f[group]["Y_left"]
            if group != ("sub"+str(args.sub)):

                X_train, X_val, y_train, y_val, rps_train, rps_val = train_test_split(X, Y_left, rps, shuffle=True)

                with h5py.File("".join([out_dir,"sub" + str(args.sub), "train"]), "a") as f2:
                    keys = f2.keys()
                    if "train" not in keys:
                        gr = f2.create_group("train")
                        gr.create_dataset('X',
                                          data=X_train,
                                          dtype='float32',
                                          maxshape=(None,))
                        gr.create_dataset('RPS',
                                          data=rps,
                                          dtype='float32',
                                          maxshape=(None,))
                        gr.create_dataset('Y_left',
                                          data=Y_left,
                                          dtype='float32',
                                          maxshape=(None,))
                    # TODO: resize and insert