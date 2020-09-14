#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""
import argparse
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split

sys.path.insert(1, r'')

from Dataset import ECoG_Dataset
from ECoG.dl.DL_utils import *
from ECoG.dl.params import Params
from ECoG.dl.tests.test_net import *
from ECoG.dl.train import train
from Models import LeNet5, SCNN_swap

#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument('--sub', type=int, default='1',
                        help="Input data directory (default= 1)")
    parser.add_argument('--finger', type=int, default='0',
                        help="Finger (default= 0)")

    # Directories
    parser.add_argument('--data_dir', type=str, default="C:\\Users\\anellim1\Develop\Thesis\BCICIV_4_mat\\",
                        help="Input data directory (default= C:\\Users\\anellim1\Develop\Thesis\BCICIV_4_mat\\)")
    parser.add_argument('--figure_dir', type=str, default='ECoG\Figures',
                        help="Figure data directory (default= ECoG\Figures)")
    parser.add_argument('--model_dir', type=str, default='ECoG\Models',
                        help="Model data directory (default= ECoG\Models\)")

    # Model Parameters
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--batch_size_valid', type=int, default=10, metavar='N',
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--batch_size_test', type=int, default=10, metavar='N',
                        help='input batch size for  (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='lr',
                        help='Learning rate (default: 0.001),')
    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.5, metavar='N',
                        help='overlap of time window (default: 0.8s)')
    parser.add_argument('--patience', type=int, default=20, metavar='N',
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')

    args = parser.parse_args()

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    # data_dir  = os.environ['DATA_PATH']
    # Define data, model and figure path
    file_name = "sub" + str(args.sub) + "_comp.mat"
    sampling_rate = 1000

    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    # set parameters
    parameters = Params(dataset="ECoG",
                        subject_n=args.sub,
                        finger=args.finger,
                        batch_size=args.batch_size,
                        valid_batch_size=args.batch_size_valid,
                        test_batch_size=args.batch_size_test,
                        epochs=args.epochs,
                        lr=args.learning_rate,
                        duration=args.duration,
                        overlap=args.overlap,
                        patience=args.patience,
                        device=device,
                        sampling_rate=sampling_rate)

    # Import dataset, split in train and tests
    dataset = ECoG_Dataset(data_dir, file_name, parameters.finger, parameters.duration, parameters.sampling_rate,
                           parameters.overlap)

    train_len, valid_len, test_len = len_split(len(dataset))
    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True, num_workers=1)
    validloader = DataLoader(valid_test, batch_size=parameters.valid_batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=True, num_workers=1)

    # net = LeNet5(in_channel=62, n_times=1000)
    net = SCNN_swap()
    print(net)

    # Training loop or model loading
    if not skip_training:
        print("Begin training...")

        optimizer = Adam(net.parameters(), lr=parameters.lr)
        loss_function = torch.nn.MSELoss()

        net, train_loss, valid_loss = train(net, trainloader, validloader, optimizer, loss_function,
                                            parameters.device, parameters.epochs, parameters.patience, model_path)

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
        plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5)  # consistent scale
        plt.xlim(0, len(train_loss) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(figure_path, "loss_plot.png"))

    if not skip_training:
        # Save the trained model
        save_pytorch_model(net, model_path, "Baselinemodel_lenet5.pth")
    else:
        # Load the model
        net = SCNN_swap()
        net = load_pytorch_model(net, os.path.join(model_path, "Baselinemodel_lenet5.pth"), "cpu")

    # Evaluation
    print("Evaluation...")
    net.eval()
    y_pred = []
    y = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(parameters.device), labels.to(parameters.device)
            y.extend(list(labels))
            y_pred.extend((list(net(data))))

    # Calculate Evaluation measures
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    print("mean squared error {}".format(mse))
    print("mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_pred))
    ax.plot(times, y_pred, color="b", label="Predicted")
    ax.plot(times, y, color="r", label="True")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Finger Movement")
    ax.set_title("SPoC Finger Movement")
    plt.legend()
    plt.savefig(os.path.join(figure_path, "DL_Finger_Prediction_LeNet5.pdf"))
    plt.show()

    # log the model
    with mlflow.start_run(experiment_id=args.experimet) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)

        mlflow.log_artifact(os.path.join(figure_path, "DL_Finger_Prediction_LeNet5.pdf"
                                         .format(mse)))
        mlflow.log_artifact(os.path.join(figure_path, "loss_plot.png"))
        mlflow.pytorch.log_model(net, "models")



