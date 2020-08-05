import torch.nn as nn
import torch.nn.functional as F


class Flatten_ECog(nn.Module):
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
            Flatten_ECog(),
            nn.Linear(32 * round(n_times / 4), 120),
            nn.ReLU(),
            nn.Linear(120, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):

        return self.net(x).squeeze(1)
