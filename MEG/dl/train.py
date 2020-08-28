from tqdm import tqdm
import numpy as np
import torch

# TODO proper citation

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def train(net, trainloader, validloader, optimizer, loss_function, device,  EPOCHS=100, patient=20 ):

    net = net.to(device) # TODO probably to remove
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        net.train()
        train_losses = []
        valid_losses = []
        for data, labels in trainloader:
            # Set data to appropiate device
            data, labels = data.to(device), labels.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Fit the network
            out = net(data)
            # Loss function
            train_loss = loss_function(out, labels[:, 0])
            train_losses.append(train_loss.item())
            # Backpropagation and weights update
            train_loss.backward()
            optimizer.step()

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, labels in validloader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(data)
                # calculate the loss
                valid_loss = loss_function(output, labels[:, 0])
                # record validation loss
                valid_losses.append(valid_loss.item())

        print("Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}"
              .format(epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)))

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

    return net, avg_train_losses, avg_valid_losses
