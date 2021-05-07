#!/usr/bin/env python

"""
    Script to test the data file created.
"""

import h5py
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=str,
        default="Z:\Desktop\\",
        help="Input data directory (default= Z:\Desktop\\)",
    )

    args = parser.parse_args()

    out_dir = args.out_dir

    print("before open the file")

    # for sub in [9]:

    # 	print("processing sub: ", sub)

    # 	with h5py.File("".join([out_dir, "data_final.hdf5"]), "r") as f:
    # 		X = f["sub"+ str(sub)]["MEG"][...]
    # 		print("MEG data shape:", X.shape)

    # 		accelermoters = f["sub"+ str(sub) + "/ACC_original"][...]
    # 		print("RPS data shape:", accelermoters.shape)

    # 		bp = f["sub"+ str(sub) + "/RPS"][...]
    # 		print("RPS data shape:", bp.shape)

    # 		y_left = f["sub"+ str(sub) + "/Y_left"][...]
    # 		print("Target data shape:", y_left.shape)

    # 	with h5py.File("".join([out_dir,"data_f.hdf5"]), "a") as f:
    # 	    grp1 = f.create_group("".join(["sub" + str(sub)]))
    # 	    # if group already exist
    # 	    # grp1 = f["".join(["sub" + str(args.sub)])]
    # 	    grp1.create_dataset("MEG", data=X, dtype='f')
    # 	    grp1.create_dataset("ACC_original", data=accelermoters, dtype='f')
    # 	    grp1.create_dataset("Y_left", data=y_left, dtype='f')
    # 	    grp1.create_dataset("RPS", data=bp, dtype='f')

    with h5py.File("".join([out_dir, "data.hdf5"]), "r") as f:
        print(f)
        print(f.keys())
        for group in f.keys():
            print("/{}".format(group))
            for dset in f[group].keys():
                print("{}{}/{}".format(f.name, group, dset))

        X = f["sub9"]["MEG"]
        print("MEG data shape:", X.shape)

        rps = f["sub9/RPS"]
        print("RPS data shape:", rps.shape)

        y_left = f.get("sub9/Y_left")
        print("Target data shape:", y_left.shape)

        y_right = f.get("sub9/Y_right")
        print("Target data shape:", y_right.shape)
