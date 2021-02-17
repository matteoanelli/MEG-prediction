#!/usr/bin/env python

"""
    Script to test the data file created.
"""

import h5py
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--out_dir', type=str, default='Z:\Desktop\\',
		help="Input data directory (default= Z:\Desktop\\)")

	args = parser.parse_args()

	out_dir = args.out_dir

	print("before open the file")

	with h5py.File("".join([out_dir, "data.hdf5"]), "r") as f:
		print(f)
		print(f.keys())
		for group in f.keys():
			print("/{}".format(group))
			for dset in f[group].keys():
				print("{}{}/{}".format(f.name, group, dset))


		X = f["sub1"]["MEG"]
		print("MEG data shape:", X.shape)

		rps = f["sub1/RPS"]
		print("RPS data shape:", rps.shape)

		y_left = f.get("sub1/Y_left")
		print("Target data shape:", y_left.shape)

		y_right = f.get("sub1/Y_right")
		print("Target data shape:", y_right.shape)
