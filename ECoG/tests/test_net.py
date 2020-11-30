import sys
import pytest
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

sys.path.insert(1, r'')

from ECoG.Utils.utils import *
from ECoG.dl.Dataset import ECoG_Dataset
from ECoG.dl.Models import LeNet5_ECoG, SCNN_ECoG

from ECoG.dl.train import train



# @pytest.mark.skip(reason="This function needs the input data")
# def test_LeNet5_shape(x, in_channel=62, n_times=500, device="cpu"):
#
#     net = LeNet5(in_channel, n_times).float()
#     net = net.to(device)
#
#     with torch.no_grad():
#         print("Shape of the input tensor: {}".format(x.shape))
#
#         y = net(x.to(device))
#         assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)
#
#     print("Test LeNet5 output shape: Success.")
#
#
# def test_LeNet5_shape2():
#
#     sample = torch.zeros([10, 62, 500])
#     net = LeNet5().float()
#
#     with torch.no_grad():
#         print("Shape of the input tensor: {}".format(sample.shape))
#
#         y = net(sample)
#         assert y.shape == torch.Size([sample.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)
#
#     print("Test LeNet5 output shape: Success.")
#
#
# def test_window_stack_shape():
#     x = torch.zeros([2, 12])
#     window = 2
#     overlap = 1
#     sample_rate = 1
#
#     x_win = window_stack(x, window, overlap, sample_rate)
#
#     assert x_win.shape == torch.Size(
#         [2, 23]
#     ), "Windowing function generating not expected shape. Expected: {}" ", Generated {}".format(x.shape, x_win.shape)
#
#     print("Test Windowing output shape: Success.")
#
#
# def test_window_stack():
#     # TODO think if the last value of the returned tensor gives any problem
#     x = torch.arange(6).reshape([1, 6])  # tensor shape [2, 3]
#     window = 2
#     overlap = 1  # stride = window - overlap
#     sample_rate = 1
#
#     x_exp = torch.Tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).unsqueeze(0)  # Tensor pre windowing
#     x_win = window_stack(x, window, overlap, sample_rate)
#
#     print("x wind:\n", x_win)
#     print("expected:\n", x_exp)
#     assert torch.equal(x_win.float(), x_exp.float()), "The windowed X does not match the expected value."
#     print("Success")
#
#
# def test_import_ECoG_tensor():
#     pass
#
#
# def test_downsampling_shape():
#
#     x = torch.ones([10, 12, 500])
#     downsampling_factor = 4
#
#     x_down = downsampling(x, down=downsampling_factor)
#     print("X original shape {}".format(x.shape))
#     print("downsampling factor = {}".format(downsampling_factor))
#
#     assert x_down.shape == torch.Size(
#         [10, 12, round(x.shape[-1] / downsampling_factor)]
#     ), "Bad shape of x downsampled. Expected shape {}, instead got {}".format(
#         torch.Size([10, 12, round(x.shape[-1] / downsampling_factor)]), x_down.shape
#     )
#
#     print("Test downsampling output shape: Success. Shape {}".format(x_down.shape))

def test_create_raw():

    X = np.zeros([100, 10000])
    y = np.ones([10000])

    raw = create_raw(X, y, 100, 1000)

    assert raw.get_data().shape == np.shape(np.zeros((101, 10000))), "Wrong shape, expected {}. Got instead {}"\
        .format(np.shape(np.zeros((101, 10000))), raw.get_data().shape)

    print("Test passed!")


def test_ECoG_dataset_shape():

    data_dir = 'C:\\Users\\anellim1\Develop\Thesis\BCICIV_4_mat\\'
    file_name = 'sub1_comp.mat'
    sampling_rate = 1000

    # test with rps
    dataset = ECoG_Dataset(data_dir, file_name, finger=0, duration=1., overlap=0.8, rps=True)

    train_len, valid_len, test_len = len_split(len(dataset))

    print(len(dataset))
    print('{} {} {}'.format(train_len, valid_len, test_len))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target, sample_bp = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 62, 501]), 'wrong data shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 1, 62, 501]), sample_data.shape)

    assert sample_target.shape == torch.Size([50]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50]), sample_target.shape)

    assert sample_bp.shape == torch.Size([50, 62, 6]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 62, 6]), sample_target.shape)

    # test without rps
    dataset = ECoG_Dataset(data_dir, file_name, finger=0, duration=1., overlap=0.8, rps=False)

    train_len, valid_len, test_len = len_split(len(dataset))

    print(len(dataset))
    print('{} {} {}'.format(train_len, valid_len, test_len))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 62, 501]), 'wrong data shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 1, 62, 501]), sample_data.shape)

    assert sample_target.shape == torch.Size([50]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50]), sample_target.shape)

    print("Test passed!")


def test_LeNet_ECoG_shape():

    x = torch.zeros([10, 1, 62, 701])
    net = LeNet5_ECoG(x.shape[-1])

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")


def test_LeNet_ECoG_train():

    train_set = TensorDataset(torch.ones([50, 1, 62, 601]), torch.zeros([50]))

    valid_set = TensorDataset(torch.ones([10, 1, 62, 601]), torch.zeros([10]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    # change between different network
    net = LeNet5_ECoG(601)
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, "")

    print('Training do not rise error')


def test_sampler():


    dataset = TensorDataset(torch.arange(100), torch.zeros([100]))

    train, test, valid = list(range(70)), list(range(70, 70+15)), list(range(70+15, 100))

    print(train)
    print(test)
    print(valid)

    train_set = Subset(dataset, train)
    test_set = Subset(dataset, test)



    print(len(train_set))

    trainloader = DataLoader(train_set, batch_size=70, shuffle=True, num_workers=1)
    testloader = DataLoader(test_set, batch_size=15, shuffle=False, num_workers=1)



    x, _ = iter(trainloader).next()
    print(x)
    x, _ = iter(testloader).next()
    print(x)


def test_SCNN_ECoG_shape():
    x = torch.zeros([10, 1, 62, 501])

    n_spatial_layer = 2
    spatial_kernel_size = [32, 31]

    temporal_n_block = 1
    # [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]]
    temporal_kernel_size = [250]
    max_pool = 2

    mlp_n_layer = 3
    mlp_hidden = 1024
    mlp_dropout = 0.5


    net = SCNN_ECoG(n_spatial_layer, spatial_kernel_size,
                              temporal_n_block, temporal_kernel_size, x.shape[-1],
                              mlp_n_layer, mlp_hidden, mlp_dropout,
                              max_pool=max_pool)
    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)

        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")


def test_SCNN_ECoG_train():

    train_set = TensorDataset(torch.ones([50, 1, 62, 501]), torch.zeros([50]))

    valid_set = TensorDataset(torch.ones([10, 1, 62, 501]), torch.zeros([10]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    n_spatial_layer = 2
    spatial_kernel_size = [32, 31]

    temporal_n_block = 1
    # [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]]
    temporal_kernel_size = [250]
    max_pool = 2

    mlp_n_layer = 3
    mlp_hidden = 1024
    mlp_dropout = 0.5

    net = SCNN_ECoG(n_spatial_layer, spatial_kernel_size,
                    temporal_n_block, temporal_kernel_size, 501,
                    mlp_n_layer, mlp_hidden, mlp_dropout,
                    max_pool=max_pool)

    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, "")

    print('Training do not rise error')