#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:38:36 2020

@author: zubarei1
"""
import mne
#from scipy import signal
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import os
#import mneflow from source save_as_numpy_branch
os.chdir('/m/nbe/project/rtmeg/problearn/mneflow')
import mneflow


data_path = "/m/nbe/scratch/strokemotor/healthy_trans/"

acc_channels_left = {1: ["MISC001", "MISC002"],
                2: ["MISC001", "MISC002"],
                3: ["MISC001", "MISC002"],
                4: ["MISC002"],
                5: ["MISC001", "MISC002"],
                6: ["MISC001", "MISC002"],
                7: ["MISC001", "MISC002"],
                8: ["MISC001", "MISC002"],
                9: ["MISC001", "MISC002"],
        }


acc_channels_right = {1: ["MISC004", "MISC005"],
                    2: ["MISC004", "MISC005"],
                    3: ["MISC005"],
                    4: ["MISC004", "MISC005"],
                    5: ["MISC004"],
                    6: ["MISC003", "MISC004"],
                    7: ["MISC003", "MISC004"],
                    8: ["MISC003", "MISC004"],
                    9: ["MISC003", "MISC004"],
                    }


def process_y(y):
    """
    Parameters:
    ----------
    y : ndarray [n_epochs, 2, n_times]


    Returns:
    --------
    y_out : ndarray [n_epochs, 1]

    """
    #center based on first 2 seconds (no movemement there)
    #y -= np.median(y[...,:250], axis=-1, keepdims=True)

    #combine directions with PCA
    if y.shape[1]==2:
        y_combined = y.transpose([1, 0, 2]).reshape([y.shape[1], -1])
        y_cov = np.cov(y_combined)
        vals, vecs = np.linalg.eig(y_cov)
        order = np.argsort(vals)[::-1]
        u = vecs[:, order][:, :1]
        #u = vecs[order, :][:1, :].T
        y_pca = np.einsum("ijk, jl -> ilk", y, u)
    else:
        y_pca = y

    #y_2 = y_pca**2#, axis=-2, keepdims=True)

    #split into segments of 256 samples(1 s) and 128ms overlap (stride=64 samples)
    #no need to use mneflow if you dont want to
    y_segmented = mneflow.utils._segment(y_pca, segment_length=250,
                                         stride=50)
    #y_out = np.squeeze(np.sqrt(np.mean(y_segmented[..., -50:]**2, axis=-1)))
    y_out = np.squeeze(np.mean(y_segmented[..., -50:], axis=-1))

    print("RMS: {} Mean: {:.2f}, range {:.2f} - {:.2f}".format(y_out.shape, y_out.mean(), y_out.min(), y_out.max()))

    #the second dimension is needed for mneflow so you can skip this
    if np.ndim(y_out) == 1:
        y_out = np.expand_dims(y_out, -1)

    return y_out


for subj_n in range(1,10):
    subj_id = "sub"+str(subj_n)+"/ball0"

    epochs = []

    raw_fnames = ["".join([data_path, subj_id, str(i), "_sss_trans.fif"]) for i in range(1,4)]
    for i, fname in enumerate(raw_fnames):
        if os.path.isfile(raw_fnames[i]):
            raw = mne.io.Raw(raw_fnames[i], preload=True)
            events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            raw.pick_types(meg='grad', misc=True)
            raw.notch_filter([50, 100])
            raw.filter(l_freq=1., h_freq=70)


            # %%
            # get indices of accelerometer channels


            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=20., decim=4))
            del raw
        else:
            print(raw_fnames[i], '***NOT FOUND')
    epochs = mne.concatenate_epochs(epochs)
    # %%
    acc_picks = mne.pick_channels(epochs.info['ch_names'],
                                                  include=acc_channels_right[subj_n])

    acc_data = epochs.get_data()[:, acc_picks, :]

    #acc_data = signal.detrend(acc_data, axis=-1)
    acc_data -= acc_data[..., :500].mean(axis=-1, keepdims=True)
    acc_data /= acc_data.std(axis=-1, keepdims=True)
    acc_data = mne.filter.filter_data(acc_data, sfreq=250, l_freq=0.5, h_freq=25.)

    meg = epochs.get_data()[:, :204, :]
    del epochs

    import_opt = dict(fs=250,
                      savepath=data_path+'//preprocessed//',
                      out_name='sub_'+str(subj_n)+'_right',
                      input_type='trials',
                      overwrite=True,
                      n_folds=5,
                      target_type='float',
                      segment=250,
                      aug_stride=50,
                      test_set = 'holdout',
                      #combine_events = {3:0, 4:1, 5:0, 6:1, 2:2},
                      scale=True,
                      scale_interval=None,
                      #decimate=2,
                      seq_length=None,
                      transform_targets = process_y,
                      scale_y=True,
                      save_as_numpy=True
                      )

    #y = process_y(acc_data_left)
    meta = mneflow.produce_tfrecords((meg, acc_data), **import_opt)
    del meg, acc_data
