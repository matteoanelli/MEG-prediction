import sys
# TODO maybe better implementation

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


class DNN_seq(nn.Module):
    def __init__(self):
        super(DNN_seq, self).__init__()

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


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.l1 = nn.Linear(204 * 1001, 2048)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 1)
        self.flatten = Flatten_MEG()

    def forward(self, x):
        x = self.flatten(x)
        x = F.dropout(F.relu(self.l1(x)), 0.5)
        x = F.dropout(F.relu(self.l2(x)), 0.5)

        return self.l3(x).squeeze(1)


class LeNet5(nn.Module):
    def __init__(self, in_channel=62, n_times=500):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, 16, 3, padding=1)
        self.max1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1, bias=False)
        self.max2 = nn.MaxPool1d(2)
        self.l1 = nn.Linear(32 * round(n_times / 4), 2048)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 1)
        self.flatten = Flatten_MEG()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = self.flatten(x)
        x = F.dropout(F.relu(self.l1(x)), 0.5)
        x = F.dropout(F.relu(self.l2(x)), 0.5)
        x = self.l3(x)

        return x.squeeze(1)


class LeNet5_seq(nn.Module):
    def __init__(self, in_channel=62, n_times=500):
        super(LeNet5_seq, self).__init__()

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

        self.spatial1 = nn.Conv2d(1, 32, kernel_size=[204, 32], bias=False)
        self.relu1 = nn.ReLU()
        self.spatial2 = nn.Conv2d(32, 64, kernel_size=[1, 32], bias=False)
        self.relu2 = nn.ReLU()
        self.spatialMax = nn.MaxPool2d(kernel_size=[1, 2])

        self.tconv1 = nn.Conv2d(1, 32, kernel_size=[16, 16], bias=False)
        self.trelu1 = nn.ReLU()
        self.tconv2 = nn.Conv2d(32, 32, kernel_size=[16, 16], bias=False)
        self.trelu2 = nn.ReLU()
        self.tmax1 = nn.MaxPool2d(kernel_size=[1, 3])
        self.tconv3 = nn.Conv2d(32, 64, kernel_size=[8, 8], bias=False)
        self.trelu3 = nn.ReLU()
        self.tconv4 = nn.Conv2d(64, 64, kernel_size=[8, 8], bias=False)
        self.trelu4 = nn.ReLU()
        self.tmax2 = nn.MaxPool2d(kernel_size=[1, 2])
        self.tconv5 = nn.Conv2d(64, 128, kernel_size=[5, 5], bias=False)
        self.trelu5 = nn.ReLU()
        self.tconv6 = nn.Conv2d(128, 128, kernel_size=[5, 5], bias=False)
        self.trelu6 = nn.ReLU()
        self.tmax3 = nn.MaxPool2d(kernel_size=[2, 2])
        self.tconv7 = nn.Conv2d(128, 128, kernel_size=[5, 5], bias=False)
        self.trelu7 = nn.ReLU()
        self.flatten = Flatten_MEG()

        self.l1 = nn.Linear(128 * 2 * 25, 1024)
        self.lrelu1 = nn.ReLU()
        self.l2 = nn.Linear(1024, 512)
        self.lrelu2 = nn.ReLU()
        self.l3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.spatial1(x)
        x = self.relu1(x)
        x = self.spatial2(x)
        x = self.relu2(x)
        x = self.spatialMax(x)
        x = torch.transpose(x, 1, 2)

        x = self.tconv1(x)
        x = self.trelu1(x)
        x = self.tconv2(x)
        x = self.trelu2(x)
        x = self.tmax1(x)
        x = self.tconv3(x)
        x = self.trelu3(x)
        x = self.tconv4(x)
        x = self.trelu4(x)
        x = self.tmax2(x)
        x = self.tconv5(x)
        x = self.trelu5(x)
        x = self.tconv6(x)
        x = self.trelu6(x)
        x = self.tmax3(x)
        x = self.tconv7(x)
        x = self.trelu7(x)

        x = self.flatten(x)
        x = self.l1(x)
        x = F.dropout(self.lrelu1(x), 0.5)
        x = self.l2(x)
        x = F.dropout(self.lrelu2(x), 0.5)
        x = self.l3(x)

        return x.squeeze(1)


class SCNN_swap_seq(nn.Module):
    def __init__(self):
        super(SCNN_swap_seq, self).__init__()

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
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, 1))

    def forward(self, x):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.flatten(x)
        x = self.ff(x)

        return x.squeeze(1)