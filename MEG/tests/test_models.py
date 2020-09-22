import os
import numpy as np
import pytest
import torch
import time
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split, TensorDataset

import MEG.dl.models as models
from MEG.Utils.utils import y_reshape, normalize, standard_scaling, y_PCA, len_split
from MEG.dl.MEG_Dataset import MEG_Dataset
from MEG.dl.train import train
from MEG.dl.hyperparameter_generation import generate_parameters, test_parameter


def test_SCNN_swap():

    net = models.SCNN_swap()

    x = torch.zeros([10, 1, 204, 1001])

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")

def test_import():
    pass

def test_y_reshaping():
    # TODO test position and velocity
    y_before = np.ones([10, 1, 1001])

    y = y_reshape(y_before, scaling=False)

    assert y.shape == (10,), "Bad shape of y with mean as measure: expected y.shape={}, got {}".format(y.shape, (10,))

    y = y_reshape(y_before, measure="movement", scaling=False)

    y_exected = np.ones([10]) * 1001.

    assert y.shape == (10,), "Bad shape of y with movement as measure: expected y.shape={}, got {}"\
        .format(y.shape, (10,))

    assert np.array_equal(y, y_exected), "Bad values of y with movement as measure: expected y: {}, got {}".format(y_exected, y)

    y_neg = y_reshape(y_before * (-1), measure="movement", scaling=False)

    assert np.array_equal(y_neg, y), "Bad values of y with movement as measure, the negative values should give the same y: " \
                       "expected {}, got {}".format(y, y_neg)


    y_before2 = np.random.rand(10, 1, 1001)

    y_mean = y_reshape(y_before2, measure="mean")
    y_movement = y_reshape(y_before2, measure="movement")
    y_vel = y_reshape(y_before2, measure="velocity")
    y_pos = y_reshape(y_before2, measure="position")

    print("y mean : {} ".format(y_mean))
    print("y movement : {} ".format(y_movement))
    print("y vel : {} ".format(y_vel))
    print("y position : {} ".format(y_pos))


def test_y_PCA():
    # Just testing the shape
    y_before = np.ones([10, 2, 1001])

    y = y_PCA(y_before)

    assert y.shape == (10, 1, 1001), "Bad shape of y: expected y.shape={}, got {}".format(y.shape, (10, 1, 1001))


def test_MEG_dataset_shape():

    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.)

    train_len, valid_len, test_len = len_split(len(dataset))

    print(len(dataset))
    print('{} {} {}'.format(train_len, valid_len, test_len))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    assert train_dataset.__len__() == 524, "Bad split, train set length expected = 524, got {}"\
        .format(train_dataset.__len__())

    assert valid_test.__len__() == 113, "Bad split, validation set length expected = 113 , got {}" \
        .format(valid_test.__len__()
                )
    assert test_dataset.__len__() == 112, "Bad split, test set length expected = 112 , got {}" \
        .format(test_dataset.__len__()
                )

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 204, 1001]), 'wrong data shape, data shape expected = {}, got {}'\
        .format(torch.Size([50, 1, 204, 1001]), sample_data.shape)

    assert sample_target.shape == torch.Size([50, 2]), 'wrong target shape, data shape expected = {}, got {}'\
        .format(torch.Size([50, 2]), sample_target.shape)



def test_normalize():
    # TODO fix add dimension
    data = torch.Tensor([[1, 1, 2, 2], [1, 1, 3, 3]]).repeat(2, 1, 1).unsqueeze(1)

    data_ = normalize(data)

    expected = torch.Tensor([[-1., -1., 0., 0.], [-1, -1, 1, 1]])\
        .repeat(2, 1, 1).unsqueeze(1)

    print("Expected = {}".format(expected))
    print("Normalized = {}".format(data_))

    assert data_.allclose(expected), "Wrong normalization!"


def test_standard_scaling():
    # TODO implement better test
    data = torch.Tensor([[1, 1, 2, 2], [1, 1, 3, 3]]).repeat(2, 1, 1).numpy()

    print(data.shape)
    print(data)

    data_mean = standard_scaling(data, log=False)
    print(data_mean)

    data_median = standard_scaling(data, scalings="median", log=False)

    expected = torch.Tensor([[-1., -1., 1., 1.], [-1, -1, 1, 1]])\
        .repeat(2, 1, 1).numpy()

    print("Expected = {}".format(expected))
    print("Stundardized = {}".format(data_mean))

    assert np.allclose(data_mean, expected), "Wrong normalization!"


@pytest.mark.skip(reason="Development porposes test")
def test_train_no_error():

    train_set = TensorDataset(torch.ones([50, 1, 204, 1001]), torch.zeros([50, 2]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 1001]), torch.zeros([10, 2]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 2

    net = models.DNN()
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs)

    print('Training do not rise error')


def test_windowing_shape():

    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.)
    # epoch lenght = 1 sec, overlap = 0.5 --> stride of 1 -0.5 --> len W_dataset 2 times datset
    windowed_dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.5)

    # the number of epoch should be doubled

    assert 2*len(dataset) == len(windowed_dataset), \
        "Something went wrong during the augmentation process, len expected: {}, got: {}".\
            format(2*len(dataset), len(windowed_dataset))


def test_len_split():
    for len in [749, 11, 12,  27, 400]:

        train, valid, test = len_split(len)

        assert len == train+valid+test, 'Splitting of the dataset wrong, total len expected: {}, got {}'\
            .format(train+valid+test, len)


def test_parameters_class():
    pass


@pytest.mark.skip(reason="Test import file")
def test_import_from_file():
    file_dir = 'Z:\Desktop\sub8\X.dat'

    print(os.path.exists(file_dir))

    start_time = time.time()
    X = np.fromfile(file_dir, dtype=float)

    print(X.shape)
    print('the X import takes: {}'.format(time.time() - start_time))

# TODO tests


def test_Spatial_Block():
    n_layer = 3
    net = models.SpatialBlock(n_layer, [104, 51, 51])
    print(net)

    x = torch.zeros([10, 1, 204, 1001])
    print(x.shape)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)

        assert y.shape == torch.Size([x.shape[0], 16*n_layer, 1, x.shape[-1]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")


def test_Temporal_Block():
    n_block = 1
    input_channel = 1
    output_channels = 64
    kernel_size = 100
    max_pool = 2

    net = models.TemporalBlock(input_channel, output_channels, kernel_size, max_pool)
    print(net)

    x = torch.zeros([10, 32, 1, 1001])
    x = torch.transpose(x, 1, 2)
    print(x.shape)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0],
                                      output_channels,
                                      x.shape[2],
                                      int((x.shape[-1] - ((kernel_size - 1) * 2 * n_block)) /
                                          max_pool if max_pool is not None else 1)])\
            ,"Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")


def test_Temporal():
    n_block = 3
    kernel_size = [100, 100, 50]
    max_pool = 2

    x = torch.zeros([10, 32, 1, 1001])
    x = torch.transpose(x, 1, 2)

    n_times_ = x.shape[-1]
    for i in range(n_block):
        n_times_ = int((n_times_ - ((kernel_size[i] - 1) * 2)))
        n_times_ = int(n_times_ / max_pool if max_pool is not None else 1)

    net = models.Temporal(n_block, kernel_size, x.shape[-1], max_pool)
    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0],
                                      n_block*16,
                                      x.shape[2],
                                      n_times_]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")


def test_MLP():
    n_times_ = 101
    n_layer = 4

    temporal_n_block = 2
    spatial_n_layer = 2
    t = temporal_n_block * 16
    c = spatial_n_layer * 16

    x = torch.zeros([10, t, c, n_times_])
    in_channel = t * c * n_times_
    print("in_channels {}".format(in_channel))

    net = models.MLP(in_channel, 516, n_layer, 0.2)
    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x.view([x.shape[0], -1]))
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")


def test_SCNN_tunable():
    x = torch.zeros([10, 1, 204, 1001])

    n_spatial_layer = 3
    spatial_kernel_size = [104, 51, 51]

    temporal_n_block = 6
    temporal_kernel_size = [20, 10, 10, 8, 8, 5]
    max_pool = 2

    mlp_n_layer = 4
    mlp_hidden = 1024
    mlp_dropout = 0.5


    net = models.SCNN_tunable(n_spatial_layer, spatial_kernel_size,
                              temporal_n_block, temporal_kernel_size, x.shape[-1],
                              mlp_n_layer, mlp_hidden, mlp_dropout,
                              max_pool=max_pool)

    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")

def test_activation():

    x = torch.arange(10, 204, 1001, dtype=float)

    act = torch.nn.ReLU()

    y = act(x)

    act_ = models.Activation("relu")

    y_ = act_(x)

    assert y.allclose(y_), "Something happen with the activation function wrapper"


def test_generate_parameters():

    param_grid = {
        "sub": [8],
        "hand": [0, 1],
        "batch_size": [80, 100, 120],
        "learning_rate": [3e-3, 4e-4],
        "duration_overlap": [(1., 0.8), (1.2, 1.), (1.4, 1.2), (0.8, 0.6), (0.6, 0.4)],
        "s_kernel_size": [[204], [54, 51, 51, 51], [104, 101], [154, 51], [104, 51, 51]],
        "t_kernel_size": [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]],
        "ff_n_layer": [1, 2, 3, 4, 5],
        "ff_hidden_channels": [1024, 516, 248],
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "activation": ["relu", "selu", "elu"]
    }

    data_dir = "data"
    model_dir = "model"
    figure_dir = "figure"

    param_sampled = generate_parameters(param_grid, 10, {"sub": 5, "activation": "relu"}, data_dir, figure_dir, model_dir)

    print(param_sampled)

def test_test_parameter():

    params= {
        "duration": 1.,
        "t_n_layer": 2,
        "t_kernel_size": [200, 50],
        "max_pool": 3
    }

    if test_parameter(params):
        print(" the parameters are ok ")
    else:
        print("Recalcolate the parameters")





