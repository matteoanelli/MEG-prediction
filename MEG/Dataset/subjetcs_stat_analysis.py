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

	data_dir = "/m/nbe/scratch/strokemotor/healthy_trans/preprocessed/"
	figure_dir = "../../pres/y_figures/"

	# print(os.path.isdir(figure_dir))


	for sub in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
		for hand in [0, 1]:
			if hand == 0:
			    file_name = "sub_{}_left.npz".format(str(sub))
			    print("##################################################")
			    print("processing file :", file_name)
			    print("##################################################")
			else:
			    file_name = "sub_{}_right.npz".format(str(sub))
			    print("##################################################")
			    print("processing file :", file_name)
			    print("##################################################")
			

			dataset = np.load(os.path.join(data_dir, file_name))

			X_train = dataset["X_train"]
			X_train = np.swapaxes(X_train, 2, -1)
			X_val = dataset["X_val"]
			X_val = np.swapaxes(X_val, 2, -1)
			X_test = dataset["X_test"]
			X_test = np.swapaxes(X_test, 2, -1)

			print("############################################################")
			print("training dataset of shape:", X_train.shape)
			print("global mean", np.mean(X_train))
			print("global min", np.min(X_train))
			print("global max", np.max(X_train))
			print("global std", np.std(X_train))

			print("############################################################")
			print("Valid dataset of shape:", X_val.shape)
			print("global mean", np.mean(X_val))
			print("global min", np.min(X_val))
			print("global max", np.max(X_val))
			print("global std", np.std(X_val))

			print("############################################################")
			print("Test dataset of shape:", X_test.shape)
			print("global mean", np.mean(X_test))
			print("global min", np.min(X_test))
			print("global max", np.max(X_test))
			print("global std", np.std(X_test))


			y_train = dataset["y_train"].squeeze()
			y_val = dataset["y_val"].squeeze()
			y_test = dataset["y_test"].squeeze()

			print("############################################################")
			print("training target of shape:", y_train.shape)
			print("global mean", np.mean(y_train))
			print("global min", np.min(y_train))
			print("global max", np.max(y_train))
			print("global std", np.std(y_train))

			print("############################################################")
			print("Valid target of shape:", y_val.shape)
			print("global mean", np.mean(y_val))
			print("global min", np.min(y_val))
			print("global max", np.max(y_val))
			print("global std", np.std(y_val))

			print("############################################################")
			print("Test target of shape:", y_test.shape)
			print("global mean", np.mean(y_test))
			print("global min", np.min(y_test))
			print("global max", np.max(y_test))
			print("global std", np.std(y_test))

			fig, axs = plt.subplots(3, figsize=[12, 6])

			axs[0].plot(y_train[:y_val.shape[0]], c='b')
			axs[0].set_ylim([np.min(y_train), np.max(y_train)])
			axs[0].set_title("Sub : {}   Train ".format(sub))

			axs[1].plot(y_val, c='b')
			axs[1].set_ylim([np.min(y_train), np.max(y_train)])
			axs[1].set_title("Val")

			axs[2].plot(y_test, c='b')
			axs[2].set_ylim([np.min(y_train), np.max(y_train)])
			axs[2].set_title("Test")
			plt.tight_layout()

			plt.savefig(os.path.join(figure_dir, "y_sub_{}_hand_{}.pdf"
			                         .format(sub, "right" if hand == 1 else "left")))














