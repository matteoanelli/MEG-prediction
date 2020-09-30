from tqdm import tqdm
import numpy as np
import torch
import os

# TODO proper citation

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
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
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(net, trainloader, validloader, optimizer, loss_function, device,  EPOCHS, patience, model_path):

    net = net.to(device) # TODO probably to remove
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(model_path, "checkpoint.pt"))

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
            train_loss = loss_function(out, labels)
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
                # Set data to appropiate device
                data, labels = data.to(device), labels.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(data)
                # calculate the loss
                valid_loss = loss_function(output, labels)
                # record validation loss
                valid_losses.append(valid_loss.item())

        print("Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}"
              .format(epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)))

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping!")
            break

    net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return net, avg_train_losses, avg_valid_losses
