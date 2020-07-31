from Models import LeNet5
import torch

def test_LeNet5_shape(x):

    net = LeNet5().float()

    with torch.no_grad():
        print('Shape of the input tensor: {}'.format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0]]), 'Bad shape of y: y.shape={}'.format(y.shape)

    print('Test LeNet5 output shape: Success.')