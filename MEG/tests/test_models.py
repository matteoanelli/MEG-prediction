import sys
import torch
import numpy as np
import pytest
import MEG.dl.models as models
from MEG.Utils.utils import y_reshape, normalize

from MEG.dl.MEG_Dataset import MEG_Dataset
from MEG.dl.train import train

from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim.adam import Adam


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

    y_before = np.zeros([10, 2, 1001])

    y = y_reshape(y_before)

    assert y.shape == (10,), "Bad shape of y: expected y.shape={}, got {}".format(y.shape, (10,))


def test_MEG_dataset_shape():

    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.)
    # test_dataset = MEG_Dataset(dataset_path, train=False, duration=1., overlap=0., test_size=0.3)

    train_dataset, valid_test, test_dataset = random_split(dataset,
                                               [
                                                    round(len(dataset)*0.5)+1,
                                                    round(len(dataset)*0.25),
                                                    round(len(dataset)*0.25)
                                                ]
                                               )

    assert train_dataset.__len__() == 375, "Bad split, train set length expected = 375, got {}"\
        .format(train_dataset.__len__())

    assert test_dataset.__len__() == 187, "Bad split, validation set length expected = 187 , got {}" \
        .format(valid_test.__len__()
                )
    assert test_dataset.__len__() == 187, "Bad split, test set length expected = 187 , got {}" \
        .format(test_dataset.__len__()
                )

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 1, 204, 1001]), 'wrong data shape, data shape expected = {}, got {}'\
        .format(torch.Size([50, 1, 204, 1001]), sample_data.shape)

    assert sample_target.shape == torch.Size([50, 2]), 'wrong target shape, data shape expected = {}, got {}'\
        .format(torch.Size([50, 2]), sample_target.shape)



def test_normalize():
    data = torch.Tensor([[1, 1, 2, 2], [1, 1, 3, 3]])

    data_ = normalize(data)

    expected = torch.Tensor([[-0.5, -0.5, 0.5, 0.5], [-1, -1, 1, 1]])

    print("Expected = {}".format(expected))
    print("Normalized = {}".format(data_))

    assert data_.allclose(expected), "Wrong normalization!"


def test_sequetial():
    data = torch.ones([10, 205, 1001])
    y = torch.ones([1001])

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




# TODO tests