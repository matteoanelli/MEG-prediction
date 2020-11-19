import sys

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, n_times):
        super(LeNet5, self).__init__()

        if n_times == 501:  # TODO automatic n_times
            self.n_times = 44 * 118
        elif n_times == 601:
            self.n_times = 44 * 143
        elif n_times == 701:
            self.n_times = 44 * 168
        else:
            raise ValueError("Network can work only with n_times = 501, 601, 701 (epoch duration of 1., 1.2, 1.4 sec),"
                             " got instead {}".format(n_times))
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            Flatten_MEG(),
            nn.Linear(64 * self.n_times, 1024),
            # nn.Linear(204 * 1001, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 120),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(120, 1),
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
class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()
    def forward(self, x, bp):

        x = x.view(x.shape[0], -1)
        bp = bp.view(bp.shape[0], -1)
        x = torch.cat([x, bp], -1)

        return x


class SCNN_tunable(nn.Module):

    def __init__(self, n_spatial_layer, spatial_kernel_size,
                 temporal_n_block, temporal_kernel_size, n_times,
                 mlp_n_layer, mlp_hidden, mlp_dropout,
                 max_pool=None, activation="relu"):
        super(SCNN_tunable, self).__init__()

        self.spatial = SpatialBlock(n_spatial_layer, spatial_kernel_size, activation)

        self.temporal = Temporal(temporal_n_block, temporal_kernel_size, n_times, activation, max_pool)

        # self.flatten = Flatten_MEG()

        self.concatenate = Concatenate()

        self.in_channel = temporal_n_block * 16 * n_spatial_layer * 16 * self.temporal.n_times_ + 204 * 6 #TODO substitue the number of channel
        self.ff = MLP(self.in_channel, mlp_hidden, mlp_n_layer, mlp_dropout, activation)

    def forward(self, x, bp):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)

        x = self.temporal(x)
        x = self.concatenate(x, bp)
        x = self.ff(x)

        return x



class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Controls the stride.
        """

        super(Block, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.Skip = True
        else:
            self.Skip = False

        self.convs = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                   nn.BatchNorm2d(out_channels)
                                   )

        self.skip_connection = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                                             nn.BatchNorm2d(out_channels))

    def forward(self, x):

        # YOUR CODE HERE
        y = self.convs(x)
        if self.Skip:
            x = self.skip_connection(x)

        return F.relu(y + x)

class GroupOfBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        super(GroupOfBlocks, self).__init__()

        first_block = Block(in_channels, out_channels, stride)
        other_blocks = [Block(out_channels, out_channels) for _ in range(1, n_blocks)]

        self.group = nn.Sequential(first_block, *other_blocks)

    def forward(self, x):

        return self.group(x)


class ResNet(nn.Module):

    def __init__(self, n_blocks, n_channels=64, n_times=501):
        """
        Args:
        n_blocks (list): A list with three elements which contains the␣

        ,→number of blocks in

        each of the three groups of blocks in ResNet.
        For instance, n_blocks = [2, 4, 6] means that the␣

        ,→first group has two blocks,

        the second group has four blocks and the third one␣

        ,→has six blocks.

        n_channels (int): Number of channels in the first group of blocks.
        num_classes (int): Number of classes.
        """
        if n_times == 501:  # TODO automatic n_times
            self.n_times = 39
        elif n_times == 601:
            self.n_times = 47
        elif n_times == 701:
            self.n_times = 55
        else:
            raise ValueError("Network can work only with n_times = 501, 601, 701 (epoch duration of 1., 1.2, 1.4 sec),"
                             " got instead {}".format(n_times))

        assert len(n_blocks) == 3, "The number of groups should be three."
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=10, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=[3, 5], stride=3, padding=1)

        self.group1 = GroupOfBlocks(n_channels, n_channels, n_blocks[0])
        self.group2 = GroupOfBlocks(n_channels, 2 * n_channels, n_blocks[1], stride=2)
        self.group3 = GroupOfBlocks(2 * n_channels, 4 * n_channels, n_blocks[2], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.flatten = Flatten_MEG()
        self.fc1 = nn.Linear(4 * n_channels * 14 * self.n_times, 516)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(516, 1)

        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, np.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x, verbose=False):
        """
        Args:
        x of shape (batch_size, 1, 28, 28): Input images.
        verbose: True if you want to print the shapes of the intermediate␣

        ,→variables.
        Returns:
        y of shape (batch_size, 10): Outputs of the network.
        """

        if verbose: print(x.shape)
        x = self.conv1(x)

        if verbose: print('conv1: ', x.shape)
        x = self.bn1(x)

        if verbose: print('bn1: ', x.shape)
        x = self.relu(x)

        if verbose: print('relu: ', x.shape)
        x = self.maxpool(x)

        if verbose: print('maxpool:', x.shape)
        x = self.group1(x)

        if verbose: print('group1: ', x.shape)
        x = self.group2(x)

        if verbose: print('group2: ', x.shape)
        x = self.group3(x)

        if verbose: print('group3: ', x.shape)
        x = self.avgpool(x)

        if verbose: print('avgpool:', x.shape)
        x = self.flatten(x)

        if verbose: print('x.view: ', x.shape)
        x = self.dropout(self.fc1(x))


        if verbose: print('fc1: ', x.shape)
        x = self.fc2(x)

        if verbose: print('out: ', x.shape)

        return x.squeeze()