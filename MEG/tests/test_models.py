"""
    Implemented models tests.
    TODO: finish tests.
"""

import pytest
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, random_split, TensorDataset

import MEG.dl.models as models
from MEG.Utils.utils import len_split
from MEG.dl.MEG_Dataset import MEG_Dataset, MEG_Dataset_no_bp
from MEG.dl.hyperparameter_generation import generate_parameters, parameter_test
from MEG.dl.train import train, train_bp, train_bp_MLP, train_2

@pytest.mark.skip(reason="To implement")
def test_Flatten_MEG():
    pass


def test_concatenate():

    x = torch.zeros(10, 5, 5)
    bp = torch.zeros(10, 204, 6)

    concatenate = models.Concatenate()

    out = concatenate(x, bp)
    expected = torch.Size([x.shape[0], 5 * 5 + 204 * 6])
    assert out.shape == expected, \
        "Wrong shape! Expected {}, got {} ".format(expected, out.shape)

    print("Test suceeeded")


def test_activation():

    x = torch.arange(10, 204, 1001, dtype=float)

    act = torch.nn.ReLU()

    y = act(x)

    act_ = models.Activation("relu")

    y_ = act_(x)

    assert y.allclose(y_), "Something happen with the activation function wrapper"


def test_LeNet_shape():

    x = torch.zeros([10, 1, 204, 701])
    net = models.LeNet5(x.shape[-1])

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")


def test_LeNet_train():

    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    # change between different network
    net = models.LeNet5(501)
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print('Training do not rise error')


def test_MNet_shape():

    x = torch.zeros([10, 1, 204, 501])

    net = models.MNet(x.shape[-1])

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")


def test_MNet_training():

    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    # change between different network
    net = models.MNet(501)
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print('Training do not rise error')


def test_RPS_MNet_shape():

    x = torch.zeros([10, 1, 204, 501])
    bp = torch.zeros([10, 204, 6])
    net = models.RPS_MNet(x.shape[-1])

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x, bp)
        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

        print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    print("Test LeNet5 output shape: Success.")


def test_RPS_MNet_training():

    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2]), torch.ones([50, 204, 6]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]), torch.ones([10, 204, 6]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    # change between different network
    net = models.RPS_MNet(501)
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train_bp(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print('Training do not rise error')


def test_Spatial_Block():
    n_layer = 3
    net = models.SpatialBlock(n_layer, [104, 51, 51], 'relu')
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
    n_block = 4
    kernel_size = [20, 10, 10, 8]
    max_pool = None

    x = torch.zeros([10, 32, 1, 501])
    x = torch.transpose(x, 1, 2)

    n_times_ = x.shape[-1]
    for i in range(n_block):
        n_times_ = int((n_times_ - ((kernel_size[i] - 1) * 2)))
        n_times_ = int(n_times_ / (max_pool if max_pool is not None else 1))

    net = models.Temporal(n_block, kernel_size, x.shape[-1], "relu", max_pool)
    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0],
                                      n_block * 16,
                                      x.shape[2],
                                      n_times_]), "Bad shape of y: y.shape={}, expected {}"\
                                                    .format(y.shape,
                                                            torch.Size(
                                                                [x.shape[0], n_block * 16, x.shape[2], n_times_]
                                                            )
                                                            )

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


def test_SCNN_shape():
    x = torch.zeros([10, 1, 204, 501])

    n_spatial_layer = 2
    spatial_kernel_size = [154, 51]

    temporal_n_block = 1
    # [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]]
    temporal_kernel_size = [250]
    max_pool = 2

    mlp_n_layer = 3
    mlp_hidden = 1024
    mlp_dropout = 0.5


    net = models.SCNN(n_spatial_layer, spatial_kernel_size,
                              temporal_n_block, temporal_kernel_size, x.shape[-1],
                              mlp_n_layer, mlp_hidden, mlp_dropout,
                              max_pool=max_pool)

    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)

        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")


def test_SCNN_training():

    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    n_spatial_layer = 2
    spatial_kernel_size = [154, 51]

    temporal_n_block = 1
    # [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]]
    temporal_kernel_size = [250]
    max_pool = 2

    mlp_n_layer = 3
    mlp_hidden = 1024
    mlp_dropout = 0.5

    net = models.SCNN(n_spatial_layer, spatial_kernel_size,
                              temporal_n_block, temporal_kernel_size, 501,
                              mlp_n_layer, mlp_hidden, mlp_dropout,
                              max_pool=max_pool)

    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _= train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print('Training do not rise error')


def test_RPS_SCNN_shape():
    x = torch.zeros([10, 1, 204, 501])
    # if RPS integration
    bp = torch.zeros([10, 204, 6])

    n_spatial_layer = 2
    spatial_kernel_size = [154, 51]

    temporal_n_block = 1
    # [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]]
    temporal_kernel_size = [250]
    max_pool = 2

    mlp_n_layer = 3
    mlp_hidden = 1024
    mlp_dropout = 0.5

    net = models.RPS_SCNN(n_spatial_layer, spatial_kernel_size,
                      temporal_n_block, temporal_kernel_size, 501,
                      mlp_n_layer, mlp_hidden, mlp_dropout,
                      max_pool=max_pool)

    print(net)

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))
        y = net(x, bp)

        assert y.shape == torch.Size([x.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test Success.")


def test_RPS_SCNN_training():
    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2]), torch.ones([50, 204, 6]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]), torch.ones([10, 204, 6]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    n_spatial_layer = 2
    spatial_kernel_size = [154, 51]

    temporal_n_block = 1
    # [[20, 10, 10, 8, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [200, 200]]
    temporal_kernel_size = [250]
    max_pool = 2

    mlp_n_layer = 3
    mlp_hidden = 1024
    mlp_dropout = 0.5

    net = models.RPS_SCNN(n_spatial_layer, spatial_kernel_size,
                      temporal_n_block, temporal_kernel_size, 501,
                      mlp_n_layer, mlp_hidden, mlp_dropout,
                      max_pool=max_pool)

    print(net)

    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train_bp(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print('Training do not rise error')


def test_Block_shape():

    # The number of channels and resolution do not change
    batch_size = 10

    x = torch.zeros(batch_size, 1, 204, 501)
    block = models.Block(in_channels=1, out_channels=1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 1, 204, 501]), "Bad shape of y: y.shape={}".format(y.shape)

    # Increase the number of channels
    block = models.Block(in_channels=1, out_channels=32)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 32, 204, 501]), "Bad shape of y: y.shape={}".format(y.shape)

    # Decrease the resolution
    block = models.Block(in_channels=1, out_channels=1, stride=2)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 1, 102, 251]), "Bad shape of y: y.shape={}".format(y.shape)

    # Increase the number of channels and decrease the resolution
    block = models.Block(in_channels=1, out_channels=32, stride=2)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 32, 102, 251]), "Bad shape of y: y.shape={}".format(y.shape)


def test_ResNet_shape():

    # Create a network with 2 block in each of the three groups
    device = "cpu"
    n_blocks = [2, 2, 2] # number of blocks in the three groups
    net = models.ResNet(n_blocks, n_channels=64, n_times=501)
    net.to(device)

    print(net)


    train_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]))
    trainloader = DataLoader(train_set, batch_size=5, shuffle=False, num_workers=1)

    with torch.no_grad():
        x, labels = iter(trainloader).next()
        x = x.to(device)
        print('Shape of the input tensor:', x.shape)

    y = net.forward(x, verbose=True)

    print(y.shape)

    assert y.shape == torch.Size([trainloader.batch_size]), "Bad shape→of y: y.shape={}".format(y.shape)

    print('Success')

# @pytest.mark.skip(reason="Development porposes test")
def test_ResNet_training():

    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    with torch.no_grad():
        x, _ = iter(trainloader).next()
    n_times = x.shape[-1]

    net = models.ResNet([2, 2, 2], 64, n_times)

    optimizer = Adam(net.parameters(), lr=0.0001)
    loss_function = torch.nn.MSELoss()

    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print("Test succeeded!")


@pytest.mark.skip(reason="Development porposes test")
def test_train_MEG_swap():

    dataset_path = ['Z:\Desktop\sub8\\ball1_sss.fif']

    dataset = MEG_Dataset(dataset_path, duration=1., overlap=0.)

    train_len, valid_len, test_len = len_split(len(dataset))

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    device = 'cpu'

    trainloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    with torch.no_grad():
        x, _, _ = iter(trainloader).next()
    n_times = x.shape[-1]

    net = models.MNet(n_times)

    optimizer = SGD(net.parameters(), lr=0.0001, weight_decay=5e-4)
    loss_function = torch.nn.MSELoss()

    model, _, _ = train(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print("Test succeeded!")


def test_generate_parameters():

    param_grid = {
        "sub": [8],
        "hand": [0, 1],
        "batch_size": [80, 100, 120],
        "learning_rate": [3e-3, 4e-4],
        "duration_overlap": [(1., 0.8), (1.2, 1.), (1.4, 1.2), (0.8, 0.6), (0.6, 0.4)],
        "s_kernel_size": [[204], [54, 51, 51, 51], [104, 101], [154, 51], [104, 51, 51]],
        "t_kernel_size": [[20, 10, 10, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [100, 75], [250]],
        "ff_n_layer": [1, 2, 3, 4, 5],
        "ff_hidden_channels": [1024, 516, 248],
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "activation": ["relu", "selu", "elu"],
        "max_pooling": [2],
    }

    data_dir = "data"
    model_dir = "model"
    figure_dir = "figure"

    param_sampled = generate_parameters(param_grid, 10, {"sub": 5, "activation": "relu"}, data_dir, figure_dir, model_dir)

    print(param_sampled)

def test_parameter():

    params= {
        "duration": 0.8,
        "t_n_layer": 2,
        "t_kernel_size": [200],
        "max_pooling": 3
    }

    if parameter_test(params):
        print(" the parameters are ok ")
    else:
        print("Recalcolate the parameters")


def test_RPS_MLP_shape():

    bp = torch.zeros([10, 204, 6])
    net = models.RPS_MLP()

    with torch.no_grad():
        print("Shape of the rps tensor: {}".format(bp.shape))

        y = net(bp)
        assert y.shape == torch.Size([bp.shape[0]]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")


def test_RPS_MLP_training():

    train_set = TensorDataset(torch.zeros([50, 204, 501]), torch.zeros([50, 2]), torch.zeros([50, 204, 6]))

    valid_set = TensorDataset(torch.zeros([10, 204, 501]), torch.zeros([10, 2]), torch.zeros([10, 204, 6]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    epochs = 1

    # change between different network
    net = models.RPS_MLP()
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train_bp_MLP(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    print('Training do not rise error')

def test_RPS_MNet_2_shape():

    x = torch.zeros([10, 1, 204, 501])
    bp = torch.zeros([10, 204, 6])
    net = models.RPS_MNet_2(x.shape[-1])

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x, bp)
        assert y.shape == torch.Size([x.shape[0], 2]), "Bad shape of y: y.shape={}".format(y.shape)

    print("Test LeNet5 output shape: Success.")


def test_RPS_MNet_2_training():
    train_set = TensorDataset(torch.ones([50, 1, 204, 501]), torch.zeros([50, 2, 2]), torch.ones([50, 204, 6]))

    valid_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2, 2]), torch.ones([10, 204, 6]))

    test_set = TensorDataset(torch.ones([10, 1, 204, 501]), torch.zeros([10, 2, 2]), torch.ones([10, 204, 6]))

    print(len(train_set))

    device = 'cpu'

    trainloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=1)

    validloader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=1)

    testloader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=1)


    epochs = 1

    with torch.no_grad():
        x, y, _ = iter(trainloader).next()
        n_times = x.shape[-1]

    # change between different network
    net = models.RPS_MNet_2(n_times)
    optimizer = Adam(net.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    print("begin training...")
    model, _, _ = train_2(net, trainloader, validloader, optimizer, loss_function, device, epochs, 10, 0, "")

    hand = 0
    model.eval()
    y_pred = []
    y = []
    with torch.no_grad():
        for data, labels, bp in testloader:
            data, labels, bp = data.to(device), labels.to(device), bp.to(device)
            y.extend(list(labels[:, hand, :]))
            y_pred.extend((list(net(data, bp))))

    y = torch.stack(y)
    y_pred = torch.stack(y_pred)
    print(y[:, 0].shape)


    print('Training do not rise error.')


def test_loss_2_param():

    hand = 0
    y_before = torch.randn([50, 2, 2])

    y = torch.randn([50, 2])

    loss_function = torch.nn.MSELoss()

    loss = loss_function(y, y_before[:, hand, :])



    assert torch.is_tensor(loss)," Something went wrong in the loss fucntion computation"

@pytest.mark.skip(reason="To implement")
def test_parameters_class():
    pass


def test_SpatialAttention():

    x = torch.zeros([10, 256, 26, 12])

    net = models.SpatialAttention()

    with torch.no_grad():
        print("Shape of the input tensor: {}".format(x.shape))

        y = net(x)
        assert y.shape == torch.Size([x.shape[0], 256, 26, 12]), "Bad shape of y: y.shape={}".format(y.shape)

        print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    print("Test LeNet5 output shape: Success.")

