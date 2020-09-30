import os, sys
import mne
import mneflow
import argparse
import tensorflow as tf

sys.path.insert(1, r'')
from MEG.Utils.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
                        help="Figure data directory (default= MEG\Figures)")
    parser.add_argument('--model_dir', type=str, default='MEG\Models',
                        help="Model data directory (default= MEG\Models\)")

    # subject
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")
    parser.add_argument('--hand', type=int, default='0',
                        help="Patient hands: 0 for sx, 1 for dx (default= 0)")

    # Epoch
    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.8, metavar='N',
                        help='overlap of time window (default: 0.8s)')

    args = parser.parse_args()

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir
    hand = args.hand
    duration = args.duration
    overlap = args.overlap
    normalize_input = True
    y_measure = "movement"

    subj_id = "/sub" + str(args.sub) + "/ball"
    # raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1 if args.sub != 3 else 2, 4)]
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]

    epochs = []
    for fname in raw_fnames:
        if os.path.exists(fname):
            raw = mne.io.Raw(raw_fnames[0], preload=True)
            # events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
            raw.pick_types(meg='grad', misc=True)
            raw.notch_filter([50, 100])
            raw.filter(l_freq=1., h_freq=70)

            # get indices of accelerometer channels
            accelerometer_picks_left = mne.pick_channels(raw.info['ch_names'],
                                                         include=["MISC001", "MISC002"])
            accelerometer_picks_right = mne.pick_channels(raw.info['ch_names'],
                                                          include=["MISC003", "MISC004"])
            epochs.append(mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=(0, 0)))
            del raw
        else:
            print("No such file '{}'".format(fname), file=sys.stderr)
    epochs = mne.concatenate_epochs(epochs)
    # get indices of accelerometer channels

    # pic only with gradiometer
    X = epochs.get_data()[:, :204, :]

    if normalize_input:
        X = standard_scaling(X, scalings="mean", log=True)

    y_left = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_left, :]), measure=y_measure)
    y_right = y_reshape(y_PCA(epochs.get_data()[:, accelerometer_picks_right, :]), measure=y_measure)

    print(
        "The input data are of shape: {}, the corresponding y_left shape is: {}," \
        "the corresponding y_right shape is: {}".format(
            X.shape, y_left.shape, y_right.shape
        )
    )

    meta = mneflow.utils.produce_tfrecords(inputs=(X, y_left if hand == 0 else y_right),
                                           savepath='../tfr/',
                                           out_name=str(hand),
                                           fs=1000,
                                           input_type="trials",
                                           target_type="signal",
                                           test_set='holdout')
    print(meta)
    dataset = mneflow.Dataset(meta, train_batch=100, test_batch=30)

    optimizer_params = dict(l2_lambda=3e-2, l1_scope=['fc'], l1_lambda=3e-4,
                            l2_scope=['dmx', 'tconv'], learn_rate=3e-4,
                            task='regression')

    optimizer = mneflow.Optimizer(**optimizer_params)

    # LFCNN model
    model_params = dict(n_ls=32,  # number of latent factors
                        filter_length=64,  # convolutional filter length
                        pooling=32,  # convlayer pooling factor
                        stride=16,  # stride parameter for pooling layer
                        padding='SAME',
                        dropout=.5,
                        nonlin=tf.nn.relu,
                        pool_type='max',
                        model_path="../tfr/",  # not used at the moment
                        )

    model = mneflow.models.LFCNN(dataset, optimizer, model_params)
    model.build()

    model.train(n_epochs=1000, eval_step=500, min_delta=1e-6, early_stopping=6)

    mneflow.utils.leave_one_subj_out()
