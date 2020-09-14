import sys

import torch.nn as nn
import torch
import torch.nn.functional as F


class Flatten_MEG(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('Inside print layer...')
        print(x.shape)
        return x


# Model only used for development purposes
class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()

        self.net = nn.Sequential(
            Flatten_MEG(),
            nn.Linear(204 * 1001, 1),
            nn.ReLU(),
            nn.Dropout(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.net = nn.Sequential(
            Flatten_MEG(),
            nn.Linear(204 * 1001, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class LeNet5(nn.Module):
    def __init__(self, in_channel=62, n_times=500):
        super(LeNet5, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channel, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.BatchNorm1d(32),
            Flatten_MEG(),
            nn.Linear(32 * round(n_times / 4), 2048),
            # nn.Linear(204 * 1001, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class SCNN_swap(nn.Module):
    def __init__(self):
        super(SCNN_swap, self).__init__()

        self.spatial = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[204, 32], bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=[1, 32], bias=False),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=[1, 2]),
                                     nn.BatchNorm2d(64))

        self.temporal = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[16, 16], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=[16, 16], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 3]),
                                      nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 64, kernel_size=[8, 8], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=[8, 8], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2]),
                                      nn.BatchNorm2d(64),
                                      nn.Conv2d(64, 128, kernel_size=[5, 5], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=[5, 5], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[2, 2]),
                                      nn.BatchNorm2d(128),
                                      nn.Conv2d(128, 128, kernel_size=[5, 5], bias=False),
                                      nn.ReLU()
                                      )

        self.concatenate = nn.Sequential()

        self.flatten = Flatten_MEG()

        self.ff = nn.Sequential(nn.Linear(128 * 2 * 25, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 1))

    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x)

        return x.squeeze(1)