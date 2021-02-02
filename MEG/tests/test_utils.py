"""
    Implemented utils tests.
    TODO: finish tests.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, random_split

from MEG.Utils.utils import y_reshape, normalize, standard_scaling, y_PCA, len_split, bandpower_1d, bandpower, \
    bandpower_multi, y_reshape_final, window_stack, standard_scaling_sklearn, len_split_cross
from MEG.dl.MEG_Dataset import MEG_Dataset, MEG_Dataset_no_bp, MEG_Dataset2


def test_bandpower_1d():
    data = np.arange(500)
    sf = 500
    band = [8, 13]

    bp = bandpower_1d(data, sf, band, relative=True)

    assert isinstance(bp, np.float64), "Something went wrong"
    print("test succeeded!")


def test_bandpower_shape():
    x = np.random.randn(10, 204, 500)
    sf = 500
    fmin = 8
    fmax = 13

    bp = bandpower(x, sf, fmin, fmax)

    assert bp.shape == np.shape(np.zeros((10, 204, 1))), \
        "Wrong shape. Expected {}, got {}".format(np.shape(np.zeros((10, 204, 1))), bp.shape)

    print("Test succeeded!")


def test_bandpower_multi_shape():
    x = np.random.randn(10, 204, 500)
    sf = 500
    bands = [(0.2, 3), (4, 7), (8, 13), (14, 31), (32, 100)]

    bp = bandpower_multi(x, sf, bands)

    assert bp.shape == np.shape(np.zeros((10, 204, len(bands)))), \
        "Wrong shape. Expected {}, got {}".format(np.shape(np.zeros((10, 204, len(bands)))), bp.shape)

    print("Test succeeded!")


@pytest.mark.skip(reason="Development porposes test")
def test_windowing_shape():
    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset_no_bp(dataset_path, duration=1., overlap=0.)

    # epoch lenght = 1 sec, overlap = 0.5 --> stride of 1 -0.5 --> len W_dataset 2 times datset
    windowed_dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.5)
    # the number of epoch should be doubled
    assert 2 * len(dataset) == len(windowed_dataset), \
        "Something went wrong during the augmentation process, len expected: {}, got: {}". \
            format(2 * len(dataset), len(windowed_dataset))


def test_window_stack_shape():
    x = torch.zeros([2, 12])
    window = 2
    overlap = 1
    sample_rate = 1

    x_win = window_stack(x, window, overlap, sample_rate)

    assert x_win.shape == torch.Size(
        [2, 23]
    ), "Windowing function generating not expected shape. Expected: {}" ", Generated {}".format(x.shape, x_win.shape)

    print("Test Windowing output shape: Success.")


def test_window_stack():
    # TODO think if the last value of the returned tensor gives any problem
    x = torch.arange(6).reshape([1, 6])  # tensor shape [2, 3]
    window = 2
    overlap = 1  # stride = window - overlap
    sample_rate = 1

    x_exp = torch.Tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).unsqueeze(0)  # Tensor pre windowing
    x_win = window_stack(x, window, overlap, sample_rate)

    print("x wind:\n", x_win)
    print("expected:\n", x_exp)
    assert torch.equal(x_win.float(), x_exp.float()), "The windowed X does not match the expected value."
    print("Success")


def test_MEG_dataset_shape():
    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset_no_bp(dataset_path, duration=1., overlap=0.)

    train_len, valid_len, test_len = len_split(len(dataset))

    print(len(dataset))
    print('{} {} {}'.format(train_len, valid_len, test_len))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    assert train_dataset.__len__() == 524, "Bad split, train set length expected = 524, got {}" \
        .format(train_dataset.__len__())

    assert valid_test.__len__() == 112, "Bad split, validation set length expected = 112 , got {}" \
        .format(valid_test.__len__()
                )
    assert test_dataset.__len__() == 113, "Bad split, test set length expected = 113 , got {}" \
        .format(test_dataset.__len__()
                )

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 204, 501]), 'wrong data shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 1, 204, 501]), sample_data.shape)

    assert sample_target.shape == torch.Size([50, 2]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 2]), sample_target.shape)


def test_MEG_dataset_shape_bp():
    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.)

    train_len, valid_len, test_len = len_split(len(dataset))

    print(len(dataset))
    print('{} {} {}'.format(train_len, valid_len, test_len))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    assert train_dataset.__len__() == 524, "Bad split, train set length expected = 524, got {}" \
        .format(train_dataset.__len__())

    assert valid_test.__len__() == 112, "Bad split, validation set length expected = 112 , got {}" \
        .format(valid_test.__len__()
                )
    assert test_dataset.__len__() == 113, "Bad split, test set length expected = 113 , got {}" \
        .format(test_dataset.__len__()
                )

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target, sample_bp = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 204, 501]), 'wrong data shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 1, 204, 501]), sample_data.shape)

    assert sample_target.shape == torch.Size([50, 2]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 2]), sample_target.shape)

    assert sample_bp.shape == torch.Size([50, 204, 6]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 204, 6]), sample_target.shape)

def test_MEG_dataset_shape_2():
    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset2(dataset_path, duration=1., overlap=0.)

    train_len, valid_len, test_len = len_split(len(dataset))

    print(len(dataset))
    print('{} {} {}'.format(train_len, valid_len, test_len))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    assert train_dataset.__len__() == 524, "Bad split, train set length expected = 524, got {}" \
        .format(train_dataset.__len__())

    assert valid_test.__len__() == 112, "Bad split, validation set length expected = 112 , got {}" \
        .format(valid_test.__len__()
                )
    assert test_dataset.__len__() == 113, "Bad split, test set length expected = 113 , got {}" \
        .format(test_dataset.__len__()
                )

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target, sample_bp = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 204, 501]), 'wrong data shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 1, 204, 501]), sample_data.shape)

    assert sample_target.shape == torch.Size([50, 2, 2]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 2, 2]), sample_target.shape)

    assert sample_bp.shape == torch.Size([50, 204, 6]), 'wrong target shape, data shape expected = {}, got {}' \
        .format(torch.Size([50, 204, 6]), sample_target.shape)

@pytest.mark.skip(reason="To implement")
def test_import_MEG_Tensor():
    # to do test with and without rps
    pass

@pytest.mark.skip(reason="To implement")
def test_import_MEG_Tensor_from_file():
    pass

@pytest.mark.skip(reason="To implement")
def test_filter_data():
    # Test filter data function, not used eventually.
    pass

@pytest.mark.skip(reason="To implement")
def test_split_data():
    # test utils.split.data
    pass


def test_y_reshaping():
    # TODO test position and velocity
    y_before = np.ones([10, 1, 1001])

    y = y_reshape(y_before, scaling=False)

    assert y.shape == (10,), "Bad shape of y with mean as measure: expected y.shape={}, got {}".format(y.shape, (10,))

    y = y_reshape(y_before, measure="movement", scaling=False)

    y_exected = np.ones([10]) * 1001.

    assert y.shape == (10,), "Bad shape of y with movement as measure: expected y.shape={}, got {}" \
        .format(y.shape, (10,))

    assert np.array_equal(y, y_exected), "Bad values of y with movement as measure: expected y: {}, got {}".format(
        y_exected, y)

    y_neg = y_reshape(y_before * (-1), measure="movement", scaling=False)

    assert np.array_equal(y_neg,
                          y), "Bad values of y with movement as measure, the negative values should give the same y: " \
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


def test_y_reshape_final():
    # test single parts
    # test PCA shaping
    y_before = np.ones([10, 2, 1001])

    y = y_PCA(y_before)

    assert y.shape == (10, 1, 1001), "Bad shape of y: expected y.shape={}, got {}".format(y.shape, (10, 1, 1001))

    # test reshapingn
    y_before = np.ones([10, 1, 1001])

    y = y_reshape(y_before, measure="movement", scaling=False)

    y_exected = np.ones([10]) * 1001.

    assert y.shape == (10,), "Bad shape of y with movement as measure: expected y.shape={}, got {}" \
        .format(y.shape, (10,))

    assert np.array_equal(y, y_exected), "Bad values of y with movement as measure: expected y: {}, got {}".format(
        y_exected, y)

    # Test standard scaling
    y_before = torch.Tensor([[1, 1, 2, 2], [1, 1, 3, 3]]).repeat(2, 1, 1).numpy()

    print(y_before.shape)
    print(y_before)

    y_mean = standard_scaling(y_before, log=False)
    print(y_mean)

    expected = torch.Tensor([[-1., -1., 1., 1.], [-1, -1, 1, 1]]).repeat(2, 1, 1).numpy()

    print("Expected = {}".format(expected))
    print("Stundardized = {}".format(y_mean))

    assert np.allclose(y_mean, expected), "Wrong normalization!"

    # Test all function shape

    y_before = np.ones([10, 2, 1001])

    y = y_reshape_final(y_before)

    assert y.shape == (10,), "Bad shape of y with movement as measure: expected y.shape={}, got {}" \
        .format((10,), y.shape)

    print("Test passed!")


def test_normalize():
    # TODO fix add dimension
    data = torch.Tensor([[1, 1, 2, 2], [1, 1, 3, 3]]).repeat(2, 1, 1).unsqueeze(1)

    data_ = normalize(data)

    expected = torch.Tensor([[-1., -1., 0., 0.], [-1, -1, 1, 1]]) \
        .repeat(2, 1, 1).unsqueeze(1)

    print("Expected = {}".format(expected))
    print("Normalized = {}".format(data_))

    assert data_.allclose(expected), "Wrong normalization!"


def test_standard_scaling():
    # TODO implement better test
    data = torch.Tensor([[1, 1, 2, 2], [1, 1, 5, 5]]).repeat(2, 1, 1).numpy()

    print(data.shape)
    print('data input: {}'.format(data))

    data_mean = standard_scaling(data, log=False)

    # ata_median = standard_scaling(data, scalings="median", log=False)

    expected = torch.Tensor([[-1., -1., 1., 1.], [-1., -1., 1., 1.]]) \
        .repeat(2, 1, 1).numpy()

    print("Expected = {}".format(expected))
    print("Stundardized = {}".format(data_mean))

    assert np.allclose(data_mean, expected), "Wrong normalization!"

    # test y shape
    data = np.random.rand(10)
    data = np.expand_dims(data, axis=1)
    print(data.shape)
    print("data: {}".format(data))
    data = standard_scaling(data)
    print("scaled data: {}".format(data))


def test_standard_skScaling():

    data = torch.Tensor([[0, 0, 0, 0], [2, 2, 2, 2]]).repeat(2, 1, 1).numpy()

    print(data.shape)
    print('data input: {}'.format(data))

    data_mean = standard_scaling_sklearn(data)

    # ata_median = standard_scaling(data, scalings="median", log=False)

    expected = torch.Tensor([[-1., -1., -1., -1.], [1., 1., 1., 1.]]) \
        .repeat(2, 1, 1).numpy()

    print("Expected = {}".format(expected))
    print("Stundardized = {}".format(data_mean))

    assert np.allclose(data_mean, expected), "Wrong normalization!"

    # test y shape
    data = np.random.rand(5, 10)
    data = np.expand_dims(data, axis=0)
    print("data shape: ", data.shape)
    scaled = standard_scaling(data)
    print("scaled data shape: {}".format(scaled.shape))



def test_len_split():
    for len in range(2000):
        train, valid, test = len_split(len)

        assert len == train + valid + test, 'Splitting of the dataset wrong, total len expected: {}, got {}' \
            .format(train + valid + test, len)


def test_len_split_cross():
    for len in range(2000):
        train, valid = len_split_cross(len)

        assert len == train + valid, 'Splitting of the dataset wrong, total len expected: {}, got {}' \
            .format(train + valid, len)

