#!/usr/bin/env python

### Script tu generate psd plot from different subjects

import os
import mne
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import iter_topography
from mne.time_frequency import psd_welch

if __name__ == "__main__":


	# Directories

	
	figure_dir = "../../pres/psd_figures/"

	print(os.path.isdir(figure_dir))

	for sub in [1, 2, 3, 4, 5, 6, 7, 8, 9]:

		data_dir = "/m/nbe/scratch/strokemotor/healthy_trans/"

		subj_id = "sub" + str(sub) + "/ball0"

		print("processing sub : ", sub)

		raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss_trans.fif"]) for i in range(1, 3)]
		print(raw_fnames)
		raws = []
		for fname in raw_fnames:
		        if os.path.isfile(fname):
		        # raw = mne.io.Raw(fname, preload=True).crop(tmax=180)
		        	raw = mne.io.Raw(fname, preload=True)
		        	print(raw.info)
		        	events = mne.find_events(
		        	    raw, stim_channel="STI101", min_duration=0.003
		        	)
		        	raw.pick_types(meg="grad", misc=False)
		        	# raw.notch_filter(50, notch_widths=2)
		        	# raw.filter(l_freq=1.0, h_freq=70)

		        	raws.append(raw)
		        	del raw
		        else:
		        	print(fname, "***NOT FOUND")

		raw = mne.concatenate_raws(raws)
		del raws
		print(len(raw))

		raw.plot_psd(show=False)
		plt.savefig(os.path.join(figure_dir, "plot_psd_sub{}_after_head_compensation.pdf"
			.format(sub)))
		del raw
			
		# before compensation head position
		data_dir = "/m/nbe/scratch/strokemotor/healthysubjects/"
		subj_id = "sub" + str(sub) + "/ball"

		raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 3)]
		raws = []
		for fname in raw_fnames:
		        if os.path.isfile(fname):
		        # raw = mne.io.Raw(fname, preload=True).crop(tmax=180)
		        	raw = mne.io.Raw(fname, preload=True)
		        	print(raw.info)
		        	events = mne.find_events(
		        	    raw, stim_channel="STI101", min_duration=0.003
		        	)
		        	raw.pick_types(meg="grad", misc=False)
		        	# raw.notch_filter(50, notch_widths=2)
		        	# raw.filter(l_freq=1.0, h_freq=70)

		        	raws.append(raw)
		        	del raw
		        else:
		        	print(fname, "***NOT FOUND")

		raw = mne.concatenate_raws(raws)
		del raws
		print(len(raw))

		raw.plot_psd(show=False)
		plt.savefig(os.path.join(figure_dir, "plot_psd_sub{}_before_head_compensation.pdf"
			.format(sub)))
		del raw


		# data before max filtering
		# subj_id = "sub" + str(sub) + "/ball"

		# raw_fnames = ["".join([data_dir, subj_id, str(i), ".fif"])
		#              for i in range(1, 2)]