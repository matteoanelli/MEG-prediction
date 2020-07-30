import torch.nn as nn
import torch.nn.functional as F

class Flatten_ECog(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(1,16,3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(16,32,3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 Flatten_ECog(),
                                 nn.Linear(15 * 175 * 32, 120),
                                 nn.ReLU(),
                                 nn.Linear(120,48),
                                 nn.ReLU(),
                                 nn.Linear(48,1))
    def forward(self, x):

        return self.net(x)
