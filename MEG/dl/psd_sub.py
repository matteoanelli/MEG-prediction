#!/usr/bin/env python
"""
    Main script to train the different models.
    This script is meant to be run to train all the different architectures tested. It receives in input data and
    architecture parameters. Such that each run of this script generate and test a new model from a specific parameters
"""
import sys
import json

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import argparse
import time as timer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split, Subset

sys.path.insert(1, r"")

from MEG.dl.train import train_PSD, train_RPS_PSD
from MEG.dl.MEG_Dataset import MEG_Within_Dataset_psd
from MEG.dl.models import (PSD_cnn, PSD_cnn_deep, PSD_cnn_spatial,
                           RPS_PSD_cnn_spatial)
from MEG.dl.params import Param_PSD

from MEG.Utils.utils import *

# Set the MNE logging to worning only.
mne.set_config("MNE_LOGGING_LEVEL", "WARNING")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    # Initialize parameters
    parameters = Param_PSD(subject_n=args.sub,
                           hand=args.hand,
                           batch_size=args.batch_size,
                           valid_batch_size=args.batch_size_valid,
                           test_batch_size=args.batch_size_test,
                           epochs=args.epochs,
                           lr=args.learning_rate,
                           wd=args.weight_decay,
                           patience=args.patience,
                           device=device,
                           batch_norm=args.batch_norm,
                           # s_kernel_size=args.s_kernel_size,  # Local
                           s_kernel_size=json.loads(
                               " ".join(args.s_kernel_size)),
                           s_drop=args.s_drop,
                           mlp_n_layer=args.mlp_n_layer,
                           mlp_hidden=args.mlp_hidden,
                           mlp_drop=args.mlp_drop,
                           desc=args.desc
                           )

    # Set if generate with RPS values or not (check network architecture used later)
    # if mlp = rps-mlp, elif rps = rps-mnet, else mnet

    use_rps = True
    print("Creating dataset")

    # Generate the custom dataset
    train_dataset = MEG_Within_Dataset_psd(data_dir, parameters.subject_n,
                                           parameters.hand, mode="train")

    test_dataset = MEG_Within_Dataset_psd(data_dir, parameters.subject_n,
                                          parameters.hand, mode="test")

    valid_dataset = MEG_Within_Dataset_psd(data_dir, parameters.subject_n,
                                           parameters.hand, mode="val")

    # split the dataset in train, test and valid sets.

    print("train set {}, val set {}, test set {}".format(len(train_dataset),
                                                         len(valid_dataset),
                                                         len(test_dataset)))

    # train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len],
    #                                                        generator=torch.Generator().manual_seed(42))
    # train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])
    # Better vizualization
    # train_valid_dataset = Subset(dataset, list(range(train_len+valid_len)))
    # test_dataset = Subset(dataset, list(range(train_len+valid_len, len(dataset))))
    #
    # train_dataset, valid_dataset = random_split(train_valid_dataset, [train_len, valid_len])

    # Initialize the dataloaders

    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size,
                             shuffle=True, num_workers=1)
    validloader = DataLoader(valid_dataset,
                             batch_size=parameters.valid_batch_size,
                             shuffle=True, num_workers=1)
    testloader = DataLoader(test_dataset,
                            batch_size=parameters.test_batch_size,
                            shuffle=False, num_workers=1)

    with torch.no_grad():
        label, psd, rps = iter(trainloader).next()
        print(psd.shape)
        print(label.shape)
        print(rps.shape)

    # Get the n_times dimension

    # net = PSD_cnn()
    if use_rps:
        net = RPS_PSD_cnn_spatial(s_kernel=parameters.s_kernel_size,
                              batch_norm=parameters.batch_norm,
                              s_dropout=parameters.s_drop,
                              mlp_layers=parameters.mlp_n_layer,
                              mlp_hidden=parameters.mlp_hidden,
                              mlp_drop=parameters.mlp_drop)
    else:
        net = PSD_cnn_spatial(s_kernel=parameters.s_kernel_size,
                              batch_norm=parameters.batch_norm,
                              s_dropout=parameters.s_drop,
                              mlp_layers=parameters.mlp_n_layer,
                              mlp_hidden=parameters.mlp_hidden,
                              mlp_drop=parameters.mlp_drop)

    print(net)
    total_params = 0
    for name, parameter in net.named_parameters():
        param = parameter.numel()
        print("param {} : {}".format(name, param if parameter.requires_grad
        else 0))
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    # Training loop or model loading
    if not skip_training:
        print("Begin training....")

        # Check the optimizer before running (different from model to model)
        # optimizer = Adam(net.parameters(), lr=parameters.lr)
        optimizer = Adam(net.parameters(), lr=parameters.lr,
                         weight_decay=parameters.wd)
        # optimizer = SGD(net.parameters(), lr=parameters.lr, momentum=0.9,
        #                  weight_decay=parameters.wd)
        # optimizer = SGD(net.parameters(), lr=parameters.lr, momentum=0.9)

        print("optimizer : ", optimizer)

        loss_function = torch.nn.MSELoss()
        # loss_function = torch.nn.L1Loss()
        print("loss : ", loss_function)
        start_time = timer.time()
        if use_rps:
            net, train_loss, valid_loss = train_RPS_PSD(
                net,
                trainloader,
                validloader,
                optimizer,
                loss_function,
                parameters.device,
                parameters.epochs,
                parameters.patience,
                parameters.hand,
                model_path,
            )
        else:
            net, train_loss, valid_loss = train_PSD(
                net,
                trainloader,
                validloader,
                optimizer,
                loss_function,
                parameters.device,
                parameters.epochs,
                parameters.patience,
                parameters.hand,
                model_path,
            )

        train_time = timer.time() - start_time
        print("Training done in {:.4f}".format(train_time))

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_loss) + 1), train_loss,
                 label='Training Loss')
        plt.plot(range(1, len(valid_loss) + 1), valid_loss,
                 label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r',
                    label='Early Stopping Checkpoint')

        plt.xlabel("epochs")
        plt.ylabel("loss")
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
        # TODO: fix init parameters (not used so far)
        net = PSD_cnn()
        net = load_pytorch_model(net, os.path.join(model_path, "model.pth"),
                                 parameters.device)

    # Evaluation
    print("Evaluation...")
    net.eval()
    y_pred = []
    y = []
    y_pred_valid = []
    y_valid = []

    # if RPS integration
    with torch.no_grad():
        if use_rps:
            for labels, psd, rps in testloader:
                labels, psd, rps = labels.to(parameters.device), psd.to(
                    parameters.device), rps.to(parameters.device)
                y.extend(list(labels[:, parameters.hand]))
                y_pred.extend((list(net(psd, rps))))

            for labels, psd, rps in validloader:
                labels, psd, rps = labels.to(parameters.device), \
                                   psd.to(parameters.device), \
                                   rps.to(parameters.device)
                y_valid.extend(list(labels[:, parameters.hand]))
                y_pred_valid.extend((list(net(psd, rps))))
        else:
            for labels, psd, _ in testloader:
                labels, psd = labels.to(parameters.device), psd.to(
                    parameters.device)
                y.extend(list(labels[:, parameters.hand]))
                y_pred.extend((list(net(psd))))

            for labels, psd, _ in validloader:
                labels, psd = labels.to(parameters.device), \
                              psd.to(parameters.device)
                y_valid.extend(list(labels[:, parameters.hand]))
                y_pred_valid.extend((list(net(psd))))

    # Calculate Evaluation measures
    print("Evaluation measures")
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    rmse_valid = mean_squared_error(y_valid, y_pred_valid, squared=False)
    r2_valid = r2_score(y_valid, y_pred_valid)
    valid_loss_last = min(valid_loss)

    print("Test set ")
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))
    print("r2 score {}".format(r2))

    print("Validation set")
    print("root mean squared error valid {}".format(rmse_valid))
    print("r2 score valid {}".format(r2_valid))
    print("last value of the validation loss: {}".format(valid_loss_last))

    # plot y_new against the true value focus on 200 timepoints
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(200)
    ax.plot(times, y_pred[0:200], color="b", label="Predicted")
    ax.plot(times, y[0:200], color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("Target")
    ax.set_title(
        "Sub {}, hand {}, Target prediction".format(
            str(parameters.subject_n),
            "sx" if parameters.hand == 0 else "dx")
    )
    plt.legend()
    plt.savefig(os.path.join(figure_path, "Times_prediction_focus.pdf"))
    plt.show()

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_pred))
    ax.plot(times, y_pred, color="b", label="Predicted")
    ax.plot(times, y, color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("Target")
    ax.set_title(
        "Sub {}, hand {}, Target prediction".format(
            str(parameters.subject_n),
            "sx" if parameters.hand == 0 else "dx")
    )
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
    ax.scatter(
        np.array(y_valid), np.array(y_pred_valid), color="b", label="Predicted"
    )
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # plt.legend()
    plt.savefig(os.path.join(figure_path, "Scatter_valid.pdf"))
    plt.show()

    # log the model and parameters using mlflow tracker
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_param("Time", train_time)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE_Valid", rmse_valid)
        mlflow.log_metric("R2_Valid", r2_valid)
        mlflow.log_metric("Valid_loss", valid_loss_last)

        mlflow.log_artifact(os.path.join(figure_path, "Times_prediction.pdf"))
        mlflow.log_artifact(
            os.path.join(figure_path, "Times_prediction_focus.pdf")
        )
        mlflow.log_artifact(os.path.join(figure_path, "loss_plot.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "Scatter.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "Scatter_valid.pdf"))
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
    parser.add_argument('--batch_size_valid', type=int, default=30,
                        metavar='N',
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--batch_size_test', type=int, default=30, metavar='N',
                        help='input batch size for  (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        metavar='lr', help='Learning rate (default: 1e-3),')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='wd', help='Weight dacay (default: 5e-4),')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='Early stopping patience (default: 20)')

    # Model architecture parameter
    parser.add_argument("--s_kernel_size", type=str, default=[204],
                        metavar="N", nargs="+", help="Spatial sub-net "
                                                     "kernel sizes (default: [104, 101])")
    parser.add_argument("--batch_norm", type=str2bool, default=False,
                        metavar="N",
                        help="Batch normalization after spatial conv layers "
                             "(default: False)", )
    parser.add_argument("--s_drop", type=str2bool, default=False, metavar="N",
                        help="Dropout after spatial conv layers "
                             "(default: False)", )
    parser.add_argument("--mlp_n_layer", type=int, default=2, metavar="N",
                        help="MLP sub-net number of layer (default: 2)")
    parser.add_argument("--mlp_hidden", type=int, default=512, metavar="N",
                        help="MLP sub-net number of hidden channels (default: 512)")
    parser.add_argument("--mlp_drop", type=float, default=0.4, metavar="d",
                        help="MLP dropout (default: 0.4),")
    # Experiment parameters
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')
    parser.add_argument('--desc', type=str, default="Normal test", metavar='N',
                        help='Experiment description (default: Normal test)')

    args = parser.parse_args()

    main(args)
