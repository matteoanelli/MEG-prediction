#!/usr/bin/env python
"""
    Transfer learning tuning.
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
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn

sys.path.insert(1, r'')

from MEG.dl.train import train, train_bp, train_bp_MLP, train_bp_transfer, train_bp_fine_tuning, train_mlp_transfer
from MEG.dl.MEG_Dataset import MEG_Dataset, MEG_Dataset_no_bp, MEG_Cross_Dataset
from MEG.dl.models import SCNN, DNN, Sample, RPS_SCNN, LeNet5, ResNet, MNet, RPS_MNet, RPS_MLP
from MEG.dl.params import Params_transf

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
    parameters = Params_transf(subject_n=args.sub,
                              hand=args.hand,
                              test_batch_size=args.batch_size_test,
                              lr=args.learning_rate,
                              device=device,
                              desc=args.desc
                              )
    # Import data and generate train-, valid- and test-set
    # Set if generate with RPS values or not (check network architecture used later)

    print("Testing: {} ".format(parameters.desc))


    leave_one_out_dataset = MEG_Cross_Dataset(data_dir, file_name, parameters.subject_n, parameters.hand, mode="test",
                                              y_measure="pca")

    # split the test set in fine_tunning and final testset
    test_len, transfer_len = len_split_cross(len(leave_one_out_dataset))

    # train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len],
    #                                                        generator=torch.Generator().manual_seed(42))

    test_dataset, transfer_dataset = random_split(leave_one_out_dataset, [test_len, transfer_len])

    # transfer_dataset = Subset(leave_one_out_dataset, list(range(transfer_len)))
    # test_dataset = Subset(leave_one_out_dataset, list(range(transfer_len, transfer_len + test_len)))

    print("Test dataset len {}, transfer dataset len {}".format(len(test_dataset), len(transfer_dataset)))

    # Initialize the dataloaders
    testloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False, num_workers=4)
    transferloader = DataLoader(transfer_dataset, batch_size=parameters.test_batch_size, shuffle=True, num_workers=4)

    # Initialize network
    # with torch.no_grad():
    #     sample, y, _ = iter(testloader).next()
    #
    # n_times = sample.shape[-1]
    # net = RPS_MNet(n_times)



    net = mlflow.pytorch.load_model("runs:/{}/models".format(args.run))

    print(net)

    # net = load_pytorch_model(net, os.path.join(model_path, "model.pth"), parameters.device)

    # Transfer learning, feature extraction.

    optimizer_trans = SGD(net.parameters(), lr=3e-4)

    loss_function_trans = torch.nn.MSELoss()
    # loss_function_trans = torch.nn.L1Loss()

    attention = False

    # net, train_loss = train_bp_transfer(net, transferloader, optimizer_trans, loss_function_trans,
    #                                     parameters.device, 100, 20,
    #                                     parameters.hand, model_path, attention)

    net, train_loss = train_bp_fine_tuning(net, transferloader, optimizer_trans, loss_function_trans,
                                             parameters.device, 100, 20,
                                             parameters.hand, model_path)


    # Evaluation
    print("Evaluation after transfer...")
    net.eval()
    y_pred = []
    y = []

    with torch.no_grad():
        for data, labels, bp in testloader:
            data, labels, bp = data.to(parameters.device), labels.to(parameters.device), bp.to(parameters.device)
            y.extend(list(labels[:, parameters.hand]))
            y_pred.extend((list(net(data, bp))))

    print("Evaluation measures")
    rmse_trans = mean_squared_error(y, y_pred, squared=False)
    r2_trans = r2_score(y, y_pred)

    print("root mean squared error after transfer learning {}".format(rmse_trans))
    print("r2 score after transfer learning  {}".format(r2_trans))

    # scatterplot y predicted against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax.scatter(np.array(y), np.array(y_pred), color="b", label="Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # plt.legend()
    plt.savefig(os.path.join(figure_path, "Scatter_after_trans.pdf"))
    plt.show()


    # log the model and parameters using mlflow tracker
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_artifact(os.path.join(figure_path, "Scatter_after_trans.pdf"))
        mlflow.pytorch.log_model(net, "models")


if __name__ == "__main__":
    # main(sys.argv[1:])

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

    # Model Parameters
    parser.add_argument('--batch_size_test', type=int, default=30, metavar='N',
                        help='input batch size for  (default: 100)')

    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='lr',
                        help='Learning rate (default: 1e-3),')

    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')
    parser.add_argument('--run', type=str, default="caadbf48e3c64647810e4945082d5f01", metavar='N',
                        help='Run to import model:')
    parser.add_argument('--desc', type=str, default="Normal test", metavar='N',
                        help='Experiment description (default: Normal test)')

    args = parser.parse_args()

    main(args)