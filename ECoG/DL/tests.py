from Models import LeNet5
import torch
from DL_utils import window_stack, downsampling


def test_LeNet5_shape(x, device='cpu'):

    net = LeNet5().float()
    net = net.to(device)

    with torch.no_grad():
        print('Shape of the input tensor: {}'.format(x.shape))

        y = net(x.to(device))
        assert y.shape == torch.Size([x.shape[0]]), 'Bad shape of y: y.shape={}'.format(y.shape)

    print('Test LeNet5 output shape: Success.')


def test_window_stack_shape():
    x = torch.zeros([2, 12])
    window = 2
    overlap = 1
    sample_rate = 1

    x_win = window_stack(x, window, overlap, sample_rate)

    assert x_win.shape == torch.Size([2, 23]), 'Windowing function generating not expected shape. Expected: {}' \
                                               ', Generated {}'.format(x.shape, x_win.shape)

    print('Test Windowing output shape: Success.')


def test_window_stack():
    # TODO think if the last value of the returned tensor gives any problem
    x = torch.arange(6).reshape([1, 6])  # tensor shape [2, 3]
    window = 2
    overlap = 1  # stride = window - overlap
    sample_rate = 1

    x_exp = torch.Tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).unsqueeze(0)  # Tensor pre windowing
    x_win = window_stack(x, window, overlap, sample_rate)

    print('x wind:\n', x_win)
    print('expected:\n', x_exp)
    assert torch.equal(x_win.float(), x_exp.float()), "The windowed X does not match the expected value."
    print('Success')


def test_import_ECoG_tensor():
    pass


def test_downsampling_shape():

    x = torch.ones([10, 12, 500])
    downsampling_factor = 4

    x_down = downsampling(x, down=downsampling_factor)
    print('X original shape {}'.format(x.shape))
    print('downsampling factor = {}'.format(downsampling_factor))

    assert x_down.shape == torch.Size([10, 12, round(x.shape[-1]/downsampling_factor)]), \
        'Bad shape of x downsampled. Expected shape {}, instead got {}'\
        .format(torch.Size([10, 12, round(x.shape[-1]/downsampling_factor)]), x_down.shape)

    print('Test downsampling output shape: Success. Shape {}'.format(x_down.shape))