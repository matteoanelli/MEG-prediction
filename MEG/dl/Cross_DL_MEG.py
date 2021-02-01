#!/usr/bin/env python
"""
    Main script to train the different models in the cross-subject analysis.
    This script is meant to be run to train all the different architectures tested.
    Such that each run of this script generate and test a new model from a specific combination of parameters.

    The approach implemented is the leave-one-out subject.
    Test set: 1 subject. (Sub in input as parameter)
    Trin set: 80% other subjects.
    Valid set: 20% other subjects.
    Meant to work only with RPS_MNET and RPS_MLP.

"""
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import argparse
import time as timer
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, random_split

sys.path.insert(1, r'')

from MEG.dl.train import train, train_bp, train_bp_MLP
from MEG.dl.MEG_Dataset import MEG_Dataset, MEG_Dataset_no_bp
from MEG.dl.models import SCNN, DNN, Sample, RPS_SCNN, LeNet5, ResNet, MNet, RPS_MNet, RPS_MLP
from MEG.dl.params import Params_tunable

from  MEG.Utils.utils import *

# Set the MNE logging to worning only.
mne.set_config("MNE_LOGGING_LEVEL", "WARNING")

def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    file_name = "data.hdf5"

    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    # Initialize parameters
    parameters = Params_tunable(subject_n=args.sub,
                                hand=args.hand,
                                batch_size=args.batch_size,
                                valid_batch_size=args.batch_size_valid,
                                test_batch_size=args.batch_size_test,
                                epochs=args.epochs,
                                lr=args.learning_rate,
                                patience=args.patience,
                                device=device,
                                y_measure=args.y_measure
                                )
    # Import data and generate train-, valid- and test-set
    # Set if generate with RPS values or not (check network architecture used later)

    mlp = False

    # Generate the custom dataset


    # split the dataset in train, test and valid sets.
    train_len, valid_len, test_len = len_split(len(dataset))
    print('{} + {} + {} = {}?'.format(train_len, valid_len, test_len, len(dataset)))

    # train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len],
    #                                                        generator=torch.Generator().manual_seed(42))
    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    # Better vizualization
    # train_valid_dataset = Subset(dataset, list(range(train_len+valid_len)))
    # test_dataset = Subset(dataset, list(range(train_len+valid_len, len(dataset))))
    #
    # train_dataset, valid_dataset = random_split(train_valid_dataset, [train_len, valid_len])

    # Initialize the dataloaders
    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True, num_workers=1)
    validloader = DataLoader(valid_test, batch_size=parameters.valid_batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False, num_workers=1)