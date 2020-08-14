import sys
# TODO maybe better implementation

import torch.nn as nn
import torch
import torch.nn.functional as F


class Flatten_MEG(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class LeNet5(nn.Module):
    def __init__(self, in_channel=62, n_times=500):
        super(LeNet5, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channel, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            Flatten_MEG(),
            nn.Linear(32 * round(n_times / 4), 120),
            nn.ReLU(),
            nn.Linear(120, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):

        return self.net(x).squeeze(1)


class SCNN_swap(nn.Module):
    def __init__(self):
        super(SCNN_swap, self).__init__()

        self.spatial = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[204, 128]),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=[1, 128]),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=[1, 64]))

        self.temporal = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[16, 16]),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=[16, 16]),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 3]),
                                      nn.Conv2d(32, 64, kernel_size=[8, 8]),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=[8, 8]),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2]),
                                      nn.Conv2d(64, 128, kernel_size=[5, 5]),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=[5, 5]),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[2, 2]),
                                      nn.Conv2d(128, 128, kernel_size=[5, 5]),
                                      nn.ReLU()
                                      )
        self.flatten = Flatten_MEG()

        self.concatenate = nn.Sequential()

        self.ff = nn.Sequential(nn.Linear(128 * 2 * 11, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, 1))


    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x).squeeze(1)
        return x