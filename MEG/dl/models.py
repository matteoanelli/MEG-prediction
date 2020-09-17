import sys

import torch.nn as nn
import torch
import torch.nn.functional as F


class Flatten_MEG(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Print(nn.Module):
    def __init__(self, message="Inside print layer"):
        super(Print, self).__init__()
        self.message = message

    def forward(self, x):
        print(self.message)
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


class SpatialBlock(nn.Module):
    def __init__(self, n_layer, kernel_size, bias=False):
        super(SpatialBlock, self).__init__()
        self.kernel_size = kernel_size
        self.out_channel = [16 * (i + 1) for i in range(n_layer)]
        self.in_channel = [1 if i == 0 else 16 * i for i in range(n_layer)]

        print(self.kernel_size)
        print(self.out_channel)
        print(self.in_channel)
        if len(kernel_size) != n_layer:
            raise ValueError(" The number of kernel passed has to be the same as the n of layer")

        self.block = nn.Sequential(*[layer for i in range(n_layer)
                                     for layer in [nn.Conv2d(self.in_channel[i],
                                                             self.out_channel[i],
                                                             kernel_size=kernel_size[i],
                                                             bias=False),
                                                   nn.ReLU(),
                                                   nn.BatchNorm2d(self.out_channel[i])
                                                   ]
                                     ])

    def forward(self, x):
        x = self.block(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, max_pool=None):
        super(TemporalBlock, self).__init__()

        self.kernel_size = kernel_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.max_pool = max_pool

        layers = [nn.Conv2d(self.input_channel, self.output_channel, kernel_size=[1, self.kernel_size], bias=False),
                  nn.ReLU(),
                  Print(message="After first conv layer"),
                  nn.Conv2d(self.output_channel, self.output_channel, kernel_size=[1, self.kernel_size], bias=False),
                  nn.BatchNorm2d(self.output_channel),
                  Print(message="after second conv layer")
                  ]

        if self.max_pool is not None:
            layers.append(nn.MaxPool2d(kernel_size=[1, self.max_pool]))
            layers.append(Print(message="After max pooling"))

        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


class Temporal(nn.Module):
    def __init__(self, n_block, kernel_size, bias=False):
        super(Temporal, self).__init__()
        self.kernel_size = kernel_size
        self.out_channel = [16 * (i + 1) for i in range(n_block)]
        self.in_channel = [1 if i == 0 else 16 * i for i in range(n_block)]

        print(self.kernel_size)
        print(self.out_channel)
        print(self.in_channel)
        if len(kernel_size) != n_block:
            raise ValueError(" The number of kernel passed has to be the same as the n of layer")

        self.block = nn.Sequential(*[layer for i in range(n_block)
                                     for layer in [nn.Conv2d(self.in_channel[i],
                                                             self.out_channel[i],
                                                             kernel_size=kernel_size[i],
                                                             bias=False),
                                                   nn.ReLU(),
                                                   nn.BatchNorm2d(self.out_channel[i]),
                                                   ]
                                     ])

        self.temporal = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[1, 16], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=[1, 16], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 3]),
                                      nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 64, kernel_size=[1, 8], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=[1, 8], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2]),
                                      nn.BatchNorm2d(64),
                                      nn.Conv2d(64, 128, kernel_size=[1, 5], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=[1, 5], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2]),
                                      nn.BatchNorm2d(128),
                                      nn.Conv2d(128, 128, kernel_size=[1, 5], bias=False),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        x = self.block(x)
        return x


class SCNN_tunable(nn.Module):
    def __init__(self, n_spatial_layer, spatial_kernel_size):
        super(SCNN_tunable, self).__init__()

        self.spatial = SpatialBlock(n_spatial_layer, spatial_kernel_size)  # TODO maybe add a the max pooling

        self.temporal = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[1, 16], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=[1, 16], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 3]),
                                      nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 64, kernel_size=[1, 8], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=[1, 8], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2]),
                                      nn.BatchNorm2d(64),
                                      nn.Conv2d(64, 128, kernel_size=[1, 5], bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=[1, 5], bias=False),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2]),
                                      nn.BatchNorm2d(128),
                                      nn.Conv2d(128, 128, kernel_size=[1, 5], bias=False),
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
