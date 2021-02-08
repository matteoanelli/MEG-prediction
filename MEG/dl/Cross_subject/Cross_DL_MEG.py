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
import torch.nn as nn

sys.path.insert(1, r'')

from MEG.dl.train import train, train_bp, train_bp_MLP, train_bp_transfer
from MEG.dl.MEG_Dataset import MEG_Dataset, MEG_Dataset_no_bp, MEG_Cross_Dataset
from MEG.dl.models import SCNN, DNN, Sample, RPS_SCNN, LeNet5, ResNet, MNet, RPS_MNet, RPS_MLP
from MEG.dl.params import Params_cross

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
    parameters = Params_cross(subject_n=args.sub,
                                hand=args.hand,
                                batch_size=args.batch_size,
                                valid_batch_size=args.batch_size_valid,
                                test_batch_size=args.batch_size_test,
                                epochs=args.epochs,
                                lr=args.learning_rate,
                                patience=args.patience,
                                device=device,
                                y_measure=args.y_measure,
                                desc= args.desc
                                )
    # Import data and generate train-, valid- and test-set
    # Set if generate with RPS values or not (check network architecture used later)

    print("Testing: {} ".format(parameters.desc))

    mlp = False

    dataset = MEG_Cross_Dataset(data_dir, file_name, parameters.subject_n, mode="train",
                                y_measure=parameters.y_measure)
    leave_one_out_dataset = MEG_Cross_Dataset(data_dir, file_name, parameters.subject_n, mode="test",
                                     y_measure=parameters.y_measure)

    # split the dataset in train, test and valid sets.
    train_len, valid_len = len_split_cross(len(dataset))

    # split the test set in fine_tunning and final testset
    test_len, transfer_len = len_split_cross(len(leave_one_out_dataset))

    # train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len],
    #                                                        generator=torch.Generator().manual_seed(42))
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    test_dataset, transfer_dataset = random_split(leave_one_out_dataset, [test_len, transfer_len])

    print("Train dataset len {}, valid dataset len {}, test dataset len {}, transfer dataset len {}".format(
        len(train_dataset), len(valid_dataset), len(test_dataset), len(transfer_dataset)))

    # Initialize the dataloaders
    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=parameters.valid_batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False, num_workers=4)
    transferloader = DataLoader(test_dataset, batch_size=parameters.valid_batch_size, shuffle=True, num_workers=4)

    # Initialize network
    if mlp:
        net = RPS_MLP()
    else:
        # Get the n_times dimension
        with torch.no_grad():
            sample, y, _ = iter(trainloader).next()

        n_times = sample.shape[-1]
        net = RPS_MNet(n_times)

    print(net)

    # Training loop or model loading
    if not skip_training:
        print("Begin training....")

        # Check the optimizer before running (different from model to model)
        # optimizer = Adam(net.parameters(), lr=parameters.lr, weight_decay=5e-4)
        optimizer = SGD(net.parameters(), lr=parameters.lr, weight_decay=5e-4)

        loss_function = torch.nn.MSELoss()
        start_time = timer.time()

        if mlp:
            net, train_loss, valid_loss = train_bp_MLP(net, trainloader, validloader, optimizer, loss_function,
                                                       parameters.device, parameters.epochs, parameters.patience,
                                                       parameters.hand, model_path)
        else:
            net, train_loss, valid_loss = train_bp(net, trainloader, validloader, optimizer, loss_function,
                                                   parameters.device, parameters.epochs, parameters.patience,
                                                   parameters.hand, model_path)

        train_time = timer.time() - start_time
        print("Training done in {:.4f}".format(train_time))

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
        plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        # plt.ylim(0, 0.5) # consistent scale
        # plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        image1 = fig
        plt.savefig(os.path.join(figure_path, "loss_plot.pdf"))

    if not skip_training:
        # Save the trained model
        save_pytorch_model(net, model_path, "model.pth")
    else:
        # Load the model (properly select the model architecture)
        net = RPS_MNet()
        net = load_pytorch_model(net, os.path.join(model_path, "model.pth"), parameters.device)

    # Evaluation
    print("Evaluation...")
    net.eval()
    y_pred = []
    y = []
    y_pred_valid = []
    y_valid = []

    # if RPS integration
    with torch.no_grad():
        if mlp:
            for _, labels, bp in testloader:
                labels, bp = labels.to(parameters.device), bp.to(parameters.device)
                y.extend(list(labels[:, parameters.hand]))
                y_pred.extend((list(net(bp))))

            for _, labels, bp in validloader:
                labels, bp = labels.to(parameters.device), bp.to(parameters.device)
                y_valid.extend(list(labels[:, parameters.hand]))
                y_pred_valid.extend((list(net(bp))))
        else:
            for data, labels, bp in testloader:
                data, labels, bp = data.to(parameters.device), labels.to(parameters.device), bp.to(parameters.device)
                y.extend(list(labels[:, parameters.hand]))
                y_pred.extend((list(net(data, bp))))

            for data, labels, bp in validloader:
                data, labels, bp = data.to(parameters.device), labels.to(parameters.device), bp.to(parameters.device)
                y_valid.extend(list(labels[:, parameters.hand]))
                y_pred_valid.extend((list(net(data, bp))))


    # Calculate Evaluation measures
    print("Evaluation measures")
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))
    print("r2 score {}".format(r2))

    # plot y_new against the true value focus on 100 timepoints
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(200)
    ax.plot(times, y_pred[0:200], color="b", label="Predicted")
    ax.plot(times, y[0:200], color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title("Sub {}, hand {}, {} prediction".format(str(parameters.subject_n),
                                                         "sx" if parameters.hand == 0 else "dx",
                                                         parameters.y_measure))
    plt.legend()
    plt.savefig(os.path.join(figure_path, "Times_prediction_focus.pdf"))
    plt.show()

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_pred))
    ax.plot(times, y_pred, color="b", label="Predicted")
    ax.plot(times, y, color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title("Sub {}, hand {}, {} prediction".format(str(parameters.subject_n),
                                                         "sx" if parameters.hand == 0 else "dx",
                                                         parameters.y_measure))
    plt.legend()
    plt.savefig(os.path.join(figure_path, "Times_prediction.pdf"))
    plt.show()

    # scatterplot y predicted against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax.scatter(np.array(y), np.array(y_pred), color="b", label="Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # plt.legend()
    plt.savefig(os.path.join(figure_path, "Scatter.pdf"))
    plt.show()

    # scatterplot y predicted against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax.scatter(np.array(y_valid), np.array(y_pred_valid), color="b", label="Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # plt.legend()
    plt.savefig(os.path.join(figure_path, "Scatter_valid.pdf"))
    plt.show()


    # Transfer learning, feature extraction.

    optimizer_trans = SGD(net.parameters(), lr=parameters.lr)

    loss_function_trans = torch.nn.MSELoss()

    net, train_loss = train_bp_transfer(net, transferloader, optimizer_trans, loss_function_trans,
                                        parameters.device, 50, parameters.patience,
                                        parameters.hand, model_path)
    # Evaluation
    print("Evaluation after transfer...")
    net.eval()
    y_pred = []
    y = []

    # if RPS integration
    with torch.no_grad():
        if mlp:
            for _, labels, bp in testloader:
                labels, bp = labels.to(parameters.device), bp.to(parameters.device)
                y.extend(list(labels[:, parameters.hand]))
                y_pred.extend((list(net(bp))))
        else:
            for data, labels, bp in testloader:
                data, labels, bp = data.to(parameters.device), labels.to(parameters.device), bp.to(
                    parameters.device)
                y.extend(list(labels[:, parameters.hand]))
                y_pred.extend((list(net(data, bp))))

    print("Evaluation measures")
    rmse_trans = mean_squared_error(y, y_pred, squared=False)
    r2_trans = r2_score(y, y_pred)

    print("root mean squared error after transfer learning {}".format(rmse))
    print("r2 score after transfer learning  {}".format(r2))

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

        mlflow.log_param("Time", train_time)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('R2', r2)

        mlflow.log_metric('RMSE_T', rmse_trans)
        mlflow.log_metric('R2_T', r2_trans)

        mlflow.log_artifact(os.path.join(figure_path, "Times_prediction.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "Times_prediction_focus.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "loss_plot.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "Scatter.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "Scatter_valid.pdf"))
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
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch_size_valid', type=int, default=30, metavar='N',
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--batch_size_test', type=int, default=30, metavar='N',
                        help='input batch size for  (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='lr',
                        help='Learning rate (default: 1e-3),')

    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--y_measure', type=str, default="left_pca",
                        help='Y type reshaping (default: left_pca)')
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')
    parser.add_argument('--desc', type=str, default="Normal test", metavar='N',
                        help='Experiment description (default: Normal test)')

    args = parser.parse_args()

    main(args)