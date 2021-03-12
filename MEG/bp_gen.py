#!/usr/bin/env python

import os, sys
import argparse
import numpy as np

sys.path.insert(1, r'')

from  MEG.Utils.utils import *

def main(args):

    data_dir = args.data_dir

    sub = args.sub

    file_name = "sub_{}_left.npz".format(str(sub))
    print("processing file :", file_name)
    out_file = "sub_{}_left_rps.npz".format(str(sub))
    print("output_file: ", out_file)

    dataset = np.load(os.path.join(data_dir, file_name))

    X_train = dataset["X_train"]
    X_train = np.swapaxes(X_train, 2, -1)
    X_val = dataset["X_val"]
    X_val = np.swapaxes(X_val, 2, -1)
    X_test = dataset["X_test"]
    X_test = np.swapaxes(X_test, 2, -1)

    bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]

    rps_train = bandpower_multi(X_train.squeeze(), fs=250, bands=bands, relative=True)
    print("train_done")

    rps_val = bandpower_multi(X_val.squeeze(), fs=250, bands=bands, relative=True)
    print("valid_done")

    rps_test = bandpower_multi(X_test.squeeze(), fs=250, bands=bands, relative=True)
    print("test_done")

    np.savez(os.path.join(data_dir, out_file), rps_train=rps_train, rps_val=rps_val, rps_test=rps_test)



if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")

    args = parser.parse_args()

    main(args)