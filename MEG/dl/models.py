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


class Activation(nn.Module):
    def __init__(self, activation="relu"):
        super(Activation, self).__init__()

        if activation == "relu":
            self.activation = nn.ReLU()

        elif activation == "selu":
            self.activation = nn.SELU()

        elif activation == "elu":
            self.activation = nn.ELU()

        else:
            raise ValueError("activetion should be one between relu, selu or elu, got instead {}".format(activation))

    def forward(self, x):

        return self.activation(x)



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
    def __init__(self, n_times):
        super(SCNN_swap, self).__init__()
        if n_times == 501:  #TODO automatic n_times
            self.n_times = 1
        elif n_times == 601:
            self.n_times = 2
        elif n_times == 701:
            self.n_times = 4
        else:
            raise ValueError("Network can work only with n_times = 501, 601, 701 (epoch duration of 1., 1.2, 1.4 sec),"
                             " got instead {}".format(n_times))

        self.spatial = nn.Sequential(nn.Conv2d(1, 32, stride=(1, 2), kernel_size=[204, 64], bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, stride=(1, 2), kernel_size=[1, 16], bias=True),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                                     )

        self.temporal = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[8, 8], bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=[8, 8], bias=True),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[5, 3], stride=(1, 2)),
                                      nn.Conv2d(32, 64, kernel_size=[1, 4], bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=[1, 4], bias=True),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                                      nn.Conv2d(64, 128, kernel_size=[1, 2], bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=[1, 2], bias=True),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                                      nn.Conv2d(128, 256, kernel_size=[1, 2], bias=True),
                                      nn.ReLU(),
                                      )

        self.concatenate = nn.Sequential()

        self.flatten = Flatten_MEG()

        self.ff = nn.Sequential(nn.Linear(256 * 46 * self.n_times, 1024),
                                nn.BatchNorm1d(num_features=1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 1024),
                                nn.BatchNorm1d(num_features=1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 1))

    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x)

        return x.squeeze(1)


class SpatialBlock(nn.Module):
    def __init__(self, n_layer, kernel_size, activation, bias=False):
        super(SpatialBlock, self).__init__()
        self.kernel_size = kernel_size
        self.out_channel = [16 * (i + 1) for i in range(n_layer)]
        self.in_channel = [1 if i == 0 else 16 * i for i in range(n_layer)]
        self.activation = activation

        if len(kernel_size) != n_layer:
            raise ValueError(" The number of kernel passed has to be the same as the n of layer")

        self.block = nn.Sequential(*[layer for i in range(n_layer)
                                     for layer in [nn.Conv2d(self.in_channel[i],
                                                             self.out_channel[i],
                                                             kernel_size=[kernel_size[i], 1],
                                                             bias=False),
                                                   Activation(self.activation),
                                                   nn.BatchNorm2d(self.out_channel[i])
                                                   ]
                                     ])

    def forward(self, x):
        x = self.block(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, max_pool=None, activation="relu"):
        super(TemporalBlock, self).__init__()

        self.kernel_size = kernel_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.max_pool = max_pool
        self.activation = activation

        layers = [nn.Conv2d(self.input_channel, self.output_channel, kernel_size=[1, self.kernel_size], bias=False),
                  Activation(self.activation),
                  nn.Conv2d(self.output_channel, self.output_channel, kernel_size=[1, self.kernel_size], bias=False),
                  nn.BatchNorm2d(self.output_channel)
                  ]

        if self.max_pool is not None:
            layers.append(nn.MaxPool2d(kernel_size=[1, self.max_pool]))

        layers.append(Activation(self.activation))

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


class Temporal(nn.Module):
    def __init__(self, n_block, kernel_size, n_times, activation, max_pool=None, bias=False):
        """
        args:

        :param n_block: (int) number of TemporalBlock the network has
        :param kernel_size: (list) list fo value for the kernel size. Each block has is own kernel size therefore
                                len(kernel_size must be == to n_block. es: [2, 3, 4]
        :param n_times: (int) n_times correspond to the last dimension of the signals.
                                It is the dimension where the convolution will apply.
        :param max_pool: (int or None) if value, it apply max pooling 2d to each block. None there is no max-pooling.
                        #TODO different max pooling factor
        :param bias:  True if conv with bias, False otherwise.

        Note: The parameters in input such as kernel size and max pooling will define the n_times dimensionality
              reduction to n_times, therefore, the reduction factor cannot be higher that n_times.
              The dimensionality reduction can be calculated as follow:

                                n_times-((sum(kernel_size) - len(kernel_size)) * 2 * n_block))
                                            / max_pool ^ n_block if max_pool is not None else 1)

               The above formula take into consideration the reduction caused by convolution as well as caused by max
               pooling. The reduction factor has to be < of n_times
        """
        super(Temporal, self).__init__()

        if len(kernel_size) != n_block:
            raise ValueError(" The number of kernel passed has to be the same as the n of layer")

        # Calculate the n_times value after forward, it has to be >= 1
        n_times_ = n_times
        for i in range(n_block):
            n_times_ = int((n_times_ - ((kernel_size[i] - 1) * 2)))
            n_times_ = int(n_times_ / (max_pool if max_pool is not None else 1))

        if n_times_ < 1:
            raise ValueError(" The reduction factor must be < than n_times. Got reduction to {}"
                             " Check kernel_sizes dimension and maxpool".format(n_times_))

        self.n_times_ = n_times_
        self.kernel_size = kernel_size
        self.out_channel = [16 * (i + 1) for i in range(n_block)]
        self.in_channel = [1 if i == 0 else 16 * i for i in range(n_block)]
        self.activation = activation
        self.max_pool = max_pool

        self.temporal = nn.Sequential(*[TemporalBlock(self.in_channel[i],
                                                      self.out_channel[i],
                                                      self.kernel_size[i],
                                                      self.max_pool,
                                                      self.activation)
                                        for i in range(n_block)
                                        ])

    def forward(self, x):
        x = self.temporal(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_layer, dropout=0.5, activation="relu"):
        super(MLP, self).__init__()

        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.n_layer = n_layer
        self.dropout = dropout
        self.activation = activation

        layers = [nn.Linear(self.in_channel, self.hidden_channel),
                  nn.Dropout(self.dropout),
                  Activation(self.activation),
                  *[layer for i in range(n_layer) for layer in [nn.Linear(self.hidden_channel, self.hidden_channel),
                                                                nn.Dropout(self.dropout),
                                                                Activation(self.activation)]],
                  nn.Linear(self.hidden_channel, 1)
                  ]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x).squeeze()

class SCNN_tunable(nn.Module):

    def __init__(self, n_spatial_layer, spatial_kernel_size,
                 temporal_n_block, temporal_kernel_size, n_times,
                 mlp_n_layer, mlp_hidden, mlp_dropout,
                 max_pool=None, activation="relu"):
        super(SCNN_tunable, self).__init__()

        self.spatial = SpatialBlock(n_spatial_layer, spatial_kernel_size, activation)  # TODO maybe add a the max pooling

        self.temporal = Temporal(temporal_n_block, temporal_kernel_size, n_times, activation, max_pool)

        self.flatten = Flatten_MEG()

        self.in_channel = temporal_n_block * 16 * n_spatial_layer * 16 * self.temporal.n_times_  #TODO maybe not a proper way of getting new n_times
        self.ff = MLP(self.in_channel, mlp_hidden, mlp_n_layer, mlp_dropout, activation)

    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)

        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x)

        return x
