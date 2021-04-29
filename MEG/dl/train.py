"""
    Training loop.
    The early stopping class is inspired by https://github.com/Bjarten/early-stopping-pytorch
"""

import os
import sys
import random

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

sys.path.insert(1, r"")

def add_gaussian_noise(data):
    """
    Add gaussian noise to imput tensor:
    Args:
        data: data tensor in imput.

    Returns:
        data: nosed data.
    """
    # multiply to change variance ex (0.1**0.5) ()
    return data + torch.randn(data.shape) * (0.1**0.5)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool):
                If True, prints a message for each validation loss improvement.
                Default: False
            delta (float):
                Minimum change in the monitored quantity to qualify as an improvement.
                Default: 0
            path (str):
                Path for the checkpoint to be saved to.
                Default: 'checkpoint.pt'
            trace_func (function):
                trace print function.
                Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train all the DL solutions.

    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        validloader (torch.utils.data.DataLoader):
            The validation loader to load the validation set.:
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.

    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
         avg_valid_losses (list):
            List of average validation loss per epoch as the model trains.

    """

    net = net.to(device)
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for data, labels, _ in trainloader:
            # Set data to appropiate device
            if random.uniform(0, 1) <= 0.8:
                data = add_gaussian_noise(data)
            data, labels = data.to(device), labels.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(data)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, labels, _ in validloader:
                # Set data to appropiate device
                data, labels = data.to(device), labels.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(data)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, hand])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print(
            "Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses


def train_bp(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train all the DL solutions with RPS integration.
    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        validloader (torch.utils.data.DataLoader):
            The validation loader to load the validation set.:
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.
    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
         avg_valid_losses (list):
            List of average validation loss per epoch as the model trains.
    """

    net = net.to(device)
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for data, labels, bp in trainloader:
            # Set data to appropiate device
            data, labels, bp = (
                data.to(device),
                labels.to(device),
                bp.to(device),
            )
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(data, bp)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, labels, bp in validloader:
                # Set data to appropiate device
                data, labels, bp = (
                    data.to(device),
                    labels.to(device),
                    bp.to(device),
                )
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(data, bp)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, hand])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print(
            "Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses


def train_bp_MLP(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train RPS_MLP.

    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        validloader (torch.utils.data.DataLoader):
            The validation loader to load the validation set.:
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.

    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
         avg_valid_losses (list):
            List of average validation loss per epoch as the model trains.

    """

    net = net.to(device)
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for _, labels, bp in trainloader:
            # Set data to appropiate device
            labels, bp = labels.to(device), bp.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(bp)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for _, labels, bp in validloader:
                # Set data to appropiate device
                labels, bp = labels.to(device), bp.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(bp)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, hand])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print(
            "Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses


def train_2(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
    fs=500,
):

    net = net.to(device)  # TODO probably to remove
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for data, labels, bp in trainloader:
            # Set data to appropiate device
            data, labels, bp = (
                data.to(device),
                labels.to(device),
                bp.to(device),
            )
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(data, bp)
            # Loss function
            train_loss = loss_function(out, labels[:, hand, :])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, labels, bp in validloader:
                # Set data to appropiate device
                data, labels, bp = (
                    data.to(device),
                    labels.to(device),
                    bp.to(device),
                )
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(data, bp)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, hand, :])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print(
            "Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses


def train_bp_transfer(
    net,
    trainloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
    attention=True,
):
    """
    Train loop used to train the rps_mnet to transfer learning.
    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.
    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
    """
    print("Transfer learning test subject...")
    # freeze all the layers
    for param in net.parameters():
        param.requires_grad = False
    # set to true the grad of the MLP
    for param in net.ff.parameters():
        param.requires_grad = True
    if attention:
        # set to true the grad of the attention layer
        for param in net.attention.parameters():
            param.requires_grad = True

    # net.ff[8] = nn.Linear(1024, 1)
    for name, param in net.named_parameters():
        print(
            "param name: {}, requires_grad {}.".format(
                name, param.requires_grad
            )
        )

    net = net.to(device)
    avg_train_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        for data, labels, bp in trainloader:
            # Set data to appropiate device
            data, labels, bp = (
                data.to(device),
                labels.to(device),
                bp.to(device),
            )
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(data, bp)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        print(
            "Epoch: {}/{}. train_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        avg_train_losses.append(train_loss)

        early_stopping(train_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses


def train_bp_fine_tuning(
    net,
    trainloader,
    optimizer,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train the rps_mnet to use fine tuning.
    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.
    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
    """
    print("Transfer learning test subject...")

    for name, param in net.named_parameters():
        print(
            "param name: {}, requires_grad {}.".format(
                name, param.requires_grad
            )
        )

    net = net.to(device)
    avg_train_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        for data, labels, bp in trainloader:
            # Set data to appropiate device
            data, labels, bp = (
                data.to(device),
                labels.to(device),
                bp.to(device),
            )
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(data, bp)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        print(
            "Epoch: {}/{}. train_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses)
            )
        )

        train_loss = np.mean(train_losses)
        avg_train_losses.append(train_loss)

        early_stopping(train_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses


def train_mlp_transfer(
    net,
    trainloader,
    optimizer,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train the rps_mlp to transfer learning.
    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.
    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
    """
    print("Transfer learning test subject...")
    # freeze all the layers
    for param in net.parameters():
        param.requires_grad = False
    # set to true the grad of the last layer of the MLP
    for param in net.ff[4].parameters():
        param.requires_grad = True

    for name, param in net.named_parameters():
        print(
            "param name: {}, requires_grad {}.".format(
                name, param.requires_grad
            )
        )

    # net.ff[8] = nn.Linear(1024, 1)

    net = net.to(device)
    avg_train_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        for _, labels, bp in trainloader:
            # Set data to appropiate device
            labels, bp = labels.to(device), bp.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(bp)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        print(
            "Epoch: {}/{}. train_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses)
            )
        )

        train_loss = np.mean(train_losses)
        avg_train_losses.append(train_loss)

        early_stopping(train_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses


def train_PSD(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train RPS_MLP.

    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        validloader (torch.utils.data.DataLoader):
            The validation loader to load the validation set.:
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.

    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
         avg_valid_losses (list):
            List of average validation loss per epoch as the model trains.

    """

    net = net.to(device)
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for labels, psd, _ in trainloader:
            # Set data to appropiate device
            labels, psd = labels.to(device), psd.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(psd)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for labels, psd, _ in validloader:
                # Set data to appropiate device
                labels, psd = labels.to(device), psd.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(psd)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, hand])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print(
            "Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses


def train_RPS_PSD(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    loss_function,
    device,
    EPOCHS,
    patience,
    hand,
    model_path,
):
    """
    Train loop used to train RPS_PSD_cnn.

    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        validloader (torch.utils.data.DataLoader):
            The validation loader to load the validation set.:
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.

    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
         avg_valid_losses (list):
            List of average validation loss per epoch as the model trains.

    """

    net = net.to(device)
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(model_path, "checkpoint.pt"),
    )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for labels, psd, rps in trainloader:
            # Set data to appropiate device
            labels, psd, rps = labels.to(device), psd.to(device), \
                               rps.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(psd, rps)
            # Loss function
            train_loss = loss_function(out, labels[:, hand])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for labels, psd, rps in validloader:
                # Set data to appropiate device
                labels, psd, rps = labels.to(device), psd.to(device), \
                                   rps.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(psd, rps)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, hand])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print(
            "Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}".format(
                epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)
            )
        )

        print("Current Learning Rate value {}".format(
            optimizer.param_groups[0]["lr"]))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

        scheduler.step(valid_loss)

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses