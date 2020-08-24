from tqdm import tqdm
import numpy as np

def train(net, trainloader, validloader, optimizer, loss_function, device,  EPOCHS=100, patient=20 ):

    net = net.to(device)
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
        for data, target in validloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            # calculate the loss
            valid_loss = loss_function(output, target)
            # record validation loss
            valid_losses.append(valid_loss.item())

        print("Epoch: {}/{}. train_loss = {:.4f}, valid_loss = {:.4f}"
              .format(epoch, EPOCHS, np.mean(train_losses), np.mean(valid_losses)))

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        return net, avg_train_losses, avg_valid_losses
