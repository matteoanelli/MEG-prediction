import sys

import torch.nn as nn
import torch
import torch.nn.functional as F

sys.path.insert(1, r'')

from MEG.dl.models import *

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
            nn.Linear(62 * 1000, 1),
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

# LeNet-like network solution.
class LeNet5_ECoG(nn.Module):
    def __init__(self, n_times):
        """

        Args:
            n_times (int):
                n_times dimension of the input data.
        """
        super(LeNet5_ECoG, self).__init__()

        if n_times == 501:  # TODO automatic n_times
            self.n_times = 8 * 118
        elif n_times == 601:
            self.n_times = 8 * 143
        elif n_times == 701:
            self.n_times = 8 * 168
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


class SCNN_ECoG(nn.Module):
    """
        SCNN Model inspired by [Kostas at al., 10.1038/s41598-019-38612-9]

        The model can be tuned automatically and the architecture is generated based on a specific combination of
        input parameters. The model is divided in 3 main bocks:
            * The spatial block that performs spatial filtering along the channel dimension only.
            * The temporal block that performs temporal filtering along the time dimension only.
            * The MLP block that combine all the feature previously extracted to optimally predict the target.
    """

    def __init__(self, n_spatial_layer, spatial_kernel_size,
                 temporal_n_block, temporal_kernel_size, n_times,
                 mlp_n_layer, mlp_hidden, mlp_dropout,
                 max_pool=None, activation="relu"):
        """

        Args:
            n_spatial_layer (int):
                Number of spatial filters applied.
            spatial_kernel_size (list):
                List of kernel sizes. The len of it has to be the same as the number of spatial filters.
            temporal_n_block (int):
                Number of temporal block applied. Each block applies 2 temporal stacked filters.
            temporal_kernel_size (list):
                List of kernel sizes. The len of it has to be the same as the number of temporal filters.
            n_times (int):
                n_times dimension of the input data.
            mlp_n_layer (int):
                Number of linear hidden layers.
            mlp_hidden (int):
                Number of neuron that the hidden layers has to have. It is designed to be the same for all of hiiden
                layers.
            mlp_dropout (float):
                Dropout percentage to apply. 0 <= mlp_dropout <= 1.
            max_pool (int):
                Max pooling factor. Default 2.
            activation (str):
                Which activation function to apply to each trainable layer. Values in [selu, relu, elu]
        """
        super(SCNN_ECoG, self).__init__()

        self.spatial = SpatialBlock(n_spatial_layer, spatial_kernel_size, activation)

        self.temporal = Temporal(temporal_n_block, temporal_kernel_size, n_times, activation, max_pool)

        self.flatten = Flatten_MEG()

        # self.concatenate = Concatenate()

        self.in_channel = temporal_n_block * 16 * n_spatial_layer * 16 * self.temporal.n_times_ #TODO substitue the number of channel
        self.ff = MLP(self.in_channel, mlp_hidden, mlp_n_layer, mlp_dropout, activation)

    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)

        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x)

        return x