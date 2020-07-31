#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""
from DL_utils import *
from Dataset import ECoG_Dataset
from torch.utils.data import DataLoader
from tests import *
from Models import LeNet5
from torch.optim.adam import Adam
from tqdm import tqdm
#%%
if __name__ == '__main__':
    # data_dir  = os.environ['DATA_PATH']
    figure_path = 'ECoG\Figures'
    model_path = 'ECoG\Models'
    data_dir = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
    file_name = 'sub1_comp.mat'
    sampling_rate = 1000
    skip_training = False

    dataset = ECoG_Dataset(data_dir, file_name, 0, 0.5, 1000)

    trainloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=1)

    data, _ = iter(trainloader).next()

    test_LeNet5_shape(data)

    net = LeNet5()

    if not skip_training:
        EPOCHS = 100
        optimizer = Adam(net.parameters(), lr=0.001)
        net.train()
        loss_function = torch.nn.MSELoss()

        for epoch in tqdm(range(1, EPOCHS+1)):
            losses = []
            for data, labels in trainloader:
                optimizer.zero_grad()

                out = net(data)

                #Loss function
                loss = loss_function(out, labels)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            print('Epoch: {}/{}. loss = {:.4f}'.format(epoch, EPOCHS, np.mean(losses)))









# TODO create the model structure done
# TODO implement model tests done architecture, missing values
# TODO select the loss function, done pytorch MSEloss
# TODO Create training loop
# TODO Manage to work with different devices
# TODO create function to save and load model
# TODO test the workflow
# TODO add feature like filtering and downsampling

