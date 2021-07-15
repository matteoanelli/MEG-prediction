#!/usr/bin/env python
"""
    Script to test cross-subject data imporr and create the test and valid dataset.
"""

import argparse
import sys

import h5py
import numpy as np
import torch

sys.path.insert(1, r"")

if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sub",
        type=int,
        default="8",
        help="Input data directory (default= 8)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Z:\Desktop\\",
        help="Input data directory (default= Z:\Desktop\\)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    file_name = "data.hdf5"

    X_train = []
    rps_train = []
    Y_left_train = []

    with h5py.File("".join([data_dir, file_name]), "r") as f:
        subjects = f.keys()

        for sub in subjects:
            if sub == ("sub" + str(args.sub)):
                X_test = f[sub]["MEG"][...]
                rps_test = f[sub]["RPS"][...]
                Y_test = f[sub]["Y_left"][...]
            else:
                X_train.append(f[sub]["MEG"][...])
                rps_train.append(f[sub]["RPS"][...])
                Y_left_train.append(f[sub]["Y_left"][...])

    X_train = torch.from_numpy(np.concatenate(X_train))
    rps_train = torch.from_numpy(np.concatenate(rps_train))
    Y_left_train = torch.from_numpy(np.concatenate(Y_left_train))

    X_test = torch.from_numpy(X_test)
    rps_test = torch.from_numpy(rps_test)
    Y_test = torch.from_numpy(Y_test)

    print(
        "Memory size of a NumPy array:",
        X_train.element_size() * X_train.nelement(),
    )

    print(
        "The size of the training data are. X: {}, rps {}, y {}".format(
            X_train.shape, rps_train.shape, Y_left_train.shape
        )
    )
    print(
        "The size of the test data are. X: {}, rps {}, y {}".format(
            X_test.shape, rps_test.shape, Y_test.shape
        )
    )
