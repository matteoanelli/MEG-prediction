import sys
# TODO maybe better implementation

import torch.nn as nn
import torch
import torch.nn.functional as F


class Flatten_MEG(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.l1 = nn.Linear(204 * 1001, 2048, bias=False)
        self.l2 = nn.Linear(2048, 1024, bias=False)
        self.l3 = nn.Linear(1024, 1, bias=False)
        self.flatten = Flatten_MEG()

    def forward(self, x):
        x = self.flatten(x)
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))

        return self.l3(x).squeeze(1)


class LeNet5(nn.Module):
    def __init__(self, in_channel=62, n_times=500):
        super(LeNet5, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channel, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            Flatten_MEG(),
            nn.Linear(32 * round(n_times / 4), 120, bias=False),
            nn.ReLU(),
            nn.Linear(120, 48, bias=False),
            nn.ReLU(),
            nn.Linear(48, 1, bias=False),
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
        self.flatten = Flatten_MEG()

        self.concatenate = nn.Sequential()

        self.ff = nn.Sequential(nn.Linear(128 * 2 * 25, 1024, bias=False),
                                nn.ReLU(),
                                nn.Linear(1024, 512, bias=False),
                                nn.ReLU(),
                                nn.Linear(512, 1, bias=False))

    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x).squeeze(1)
        return x