import sys
import torch
import numpy as np
import MEG.dl.models as models
from MEG.Utils.utils import y_reshape

from MEG.dl.MEG_Dataset import MEG_Dataset

from torch.utils.data import DataLoader, random_split

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

    dataset = MEG_Dataset(dataset_path, train=True, duration=1., overlap=0.)
    # test_dataset = MEG_Dataset(dataset_path, train=False, duration=1., overlap=0., test_size=0.3)

    train_dataset, test_dataset = random_split(dataset, [round(len(dataset)*0.5)+1, round(len(dataset)*0.5)])

    assert train_dataset.__len__() == 375, "Bad split, train set length expected = 375, got {}"\
        .format(train_dataset.__len__())

    assert test_dataset.__len__() == 374, "Bad split, test set length expected = 374 , got {}"\
        .format(test_dataset.__len__()
                )

    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=1)

    sample_data, sample_target = iter(trainloader).next()

    assert sample_data.shape == torch.Size([50, 204, 1001]), 'wrong data shape, data shape expected = {}, got {}'\
        .format(torch.Size([50, 204, 1001]), sample_data.shape)

    assert sample_target.shape == torch.Size([50, 2]), 'wrong target shape, data shape expected = {}, got {}'\
        .format(torch.Size([50, 2]), sample_target.shape)



def test_MEG_dataset():
    pass

# TODO tests