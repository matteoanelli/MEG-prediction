#!/usr/bin/env python
import os
import mne
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mne.viz import iter_topography
from mne.time_frequency import psd_welch


def my_callback(ax, ch_idx, psds):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    freqs = np.arange(70)
    psds = np.random.randn(204, 70)
    print(ax)
    print(ch_idx)
    ax.plot(freqs, psds[ch_idx], color='red')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')

def main(args):

    data_dir = args.data_dir
    figure_dir = args.figure_dir

    sub = args.sub

    subj_id = "sub" + str(sub) + "/ball0"

    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss_trans.fif"])
                  for i in range(1, 4)]

    raws = []
    for fname in raw_fnames:
        if os.path.isfile(fname):
            raw = mne.io.Raw(fname, preload=True).crop(tmax=180)
            print(raw.info)
            events = mne.find_events(
                raw, stim_channel="STI101", min_duration=0.003
            )
            raw.pick_types(meg="grad", misc=False)
            raw.notch_filter(50, notch_widths=2)
            raw.filter(l_freq=1.0, h_freq=70)

            raws.append(raw)
            del raw
        else:
            print(fname, "***NOT FOUND")

    raw = mne.concatenate_raws(raws)

    raw.plot(start=0., duration=20., n_channels=20, events=events)
    plt.show()

    psds, freqs = mne.time_frequency.psd_welch(raw, n_per_seg=250,
                                               n_overlap=250 / 2,
                                               average='mean', fmin=1.0,
                                               fmax=70)
    psds = 20 * np.log10(psds)  # scale to DB
    fig = plt.figure(figsize=(12, 8))
    for ax, idx in iter_topography(raw.info,
                                   fig_facecolor='white',
                                   axis_facecolor='white',
                                   axis_spinecolor='white',
                                   on_pick=None,
                                   fig=fig):
        ax.plot(psds[idx], color='red')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='Z:\Desktop\\',
                        help="Figure directory (default= Z:\Desktop\\)")
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")

    args = parser.parse_args()

    main(args)
