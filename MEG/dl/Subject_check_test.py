#!/usr/bin/env python
"""
   Script to check the subject accelerometer data for cross-subject analysis.
"""

import io
import os
import argparse

import numpy as np
import mne

from matplotlib import pyplot as plt


subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9] 


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    # Generate the data input path list. Each subject has 3 runs stored in 3 different files.

    for sub in subjects:

    	if sub != 3:
    		file = "/sub"+str(sub)+"/ball01_sss_trans.fif"
    	else:
    		file = "/sub"+str(sub)+"/ball02_sss_trans.fif"

    	file_path = "".join([data_dir, file])

    	raw = mne.io.Raw(file_path, preload=True)

    	print("Sub number {} info".format(sub))
    	print(raw.info)
    	print("Sub misc channels")

    	raw.pick_types(meg=False, misc=True)

    	print(raw.info['ch_names'])

    	raw.plot(duration=320, n_channels=6, show=False)
    	plt.savefig(os.path.join(figure_path, "./sub_acc/sub_{}.pdf".format(sub)))


    	del raw


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Directories
	parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
	                    help="Input data directory (default= Z:\Desktop\\)")
	parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
	                    help="Figure data directory (default= MEG\Figures)")
	parser.add_argument('--model_dir', type=str, default='MEG\Models',
	                    help="Model data directory (default= MEG\Models\)")

	args = parser.parse_args()

	main(args)
