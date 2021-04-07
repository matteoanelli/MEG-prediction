#!/usr/bin/env python
"""
    Main script to train the different models.
    This script is meant to be run to train all the different architectures tested. It receives in input data and
    architecture parameters. Such that each run of this script generate and test a new model from a specific parameters
"""
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import argparse
import numpy as np
import time as timer
from torch.nn import functional as F
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split, Subset

sys.path.insert(1, r"")

from MEG.dl.MEG_Dataset import (MEG_Dataset, MEG_Dataset_no_bp,
                                MEG_Within_Dataset, MEG_Within_Dataset_ivan)
from MEG.dl.models import (Generator, Discriminator)
from MEG.dl.params import Params_tunable, Params_cross

from MEG.Utils.utils import *


def generator_loss(D, _fake):
    """Loss computed to train the GAN generator.

    Args:
      D: The discriminator whose forward function takes inputs of shape
      (batch_size, 1, 204, 250) and produces outputs of shape (batch_size, 1).
      _fake of shape (batch_size, 1, 204, 250): Fake signals produces by
      the generator.

    Returns:
      loss: The mean of the binary cross-entropy losses computed for all the
      samples in the batch.

    Notes:
    - Make sure that you process on the device given by `_fake.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the
        right targets.
    """
    # Considering fake as 0

    return F.binary_cross_entropy(
        D(_fake).to(_fake.device),
        torch.ones((_fake.shape[0])).to(_fake.device))


def discriminator_loss(D, _real, _fake):
    """Loss computed to train the GAN discriminator.

    Args:
      D: The discriminator.
      real of shape (batch_size,  1, 204, 250): Real signals.
      fake_images of shape (batch_size,  1, 204, 250): Fake signals produces
      by the generator.

    Returns:
      d_loss_real: The mean of the binary cross-entropy losses computed on the
            real signals.
      D_real: Mean output of the discriminator for real signals.
            This is useful for tracking convergence.
      d_loss_fake: The mean of the binary cross-entropy losses computed on the
            fake signals.
      D_fake: Mean output of the discriminator for fake signals.
            This is useful for tracking convergence.

    """
    # REAL = 1
    # FAKE = 0

    out_real = D(_real).to(_real.device)
    real_target = torch.ones(_real.shape[0]).to(_real.device)
    d_loss_real = F.binary_cross_entropy(out_real, real_target)
    D_real = out_real.mean()

    out_fake = D(_fake).to(_fake.device)
    fake_target = torch.zeros(_fake.shape[0]).to(_fake.device)
    d_loss_fake = F.binary_cross_entropy(out_fake, fake_target)
    D_fake = out_fake.mean()

    return d_loss_real, D_real, d_loss_fake, D_fake

def train_gan(netG, netD, trainloader, optimizerG, optimizerD, device, EPOCHS,
              patience, model_path, nz):
    """
    Train loop used to train all the DL solutions.

    Args:
        net (torch.nn.Module):
            The network to train.
        trainloader (torch.utils.data.DataLoader):
            The train loader to load the train set.
        validloader (torch.utils.data.DataLoader):
            The validation loader to load the validation set.:
        optimizer (torch.optim.Optimizer):
            The optimizer to be used.
        loss_function (torch.nn.Module):
        device (torch.device):
            The device where run the computation.
        EPOCHS (int):
            The maximum number of epochs.
        patience:
            The early stopping patience.
        hand:
            The processes hand. 0 = left, 1 = right.
        model_path:
            The path to save the model and the checkpoints.

    Returns:
        net (torch.nn.Module):
            The trained network.
         avg_train_losses (list):
            List of average training loss per epoch as the model trains.
         avg_valid_losses (list):
            List of average validation loss per epoch as the model trains.

    """

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(
    #     patience=patience,
    #     verbose=True,
    #     path=os.path.join(model_path, "checkpoint.pt"),
    # )

    for epoch in tqdm(range(1, EPOCHS + 1)):
        ###################
        # train the model #
        ###################
        netG.train()
        netD.train()
        lossesG = []
        lossesDF = []
        lossesDR = []
        for _real, _, _ in trainloader:
            # Set data to appropiate device
            _real = _real.to(device)
            # generate uniform input fake signals
            z = torch.randn((_real.shape[0], nz, 1, 1)).to(device)
            ###################
            #  Discriminator  #
            ###################

            optimizerD.zero_grad()

            _fake = netG(z)

            d_loss_real, D_real, d_loss_fake, D_fake = discriminator_loss(
                                                        netD, _real, _fake)

            lossesDF.append(d_loss_fake.item())
            lossesDR.append(d_loss_real.item())

            d_loss_fake.backward(retain_graph=True)
            d_loss_real.backward()

            optimizerD.step()

            ###################
            #    Generator    #
            ###################
            optimizerG.zero_grad()

            _fake = netG(z)

            lossG = generator_loss(netD, _fake)
            lossG.backward()
            lossesG.append(lossG.item())

            optimizerG.step()

        print(
            'Epoch: {}/{}. loss Gen = {:.4f}, loss Dis fake {:.4f}, loss Dis '
            'Real {:.4f} '.format(epoch, EPOCHS, np.mean(lossesG),
                                  np.mean(lossesDF), np.mean(lossesDR)))

        print("D_real : {}, D_fake {}".format(D_real, D_fake))


        # early_stopping(valid_loss, net)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping!")
        #     break

    # net.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.pt")))

    return netG, netD, np.mean(lossesG), np.mean(lossesDF), \
           np.mean(lossesDR), D_fake, D_real


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    file_name = "ball_left_mean.npz"

    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    parameters = Params_cross(subject_n=args.sub,
                              hand=args.hand,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size_valid,
                              test_batch_size=args.batch_size_test,
                              epochs=args.epochs,
                              lr=args.learning_rate,
                              wd=args.weight_decay,
                              patience=args.patience,
                              device=device,
                              y_measure=args.y_measure,
                              desc=args.desc
                              )

    # Creat train dataset

    train_dataset = MEG_Within_Dataset_ivan(data_dir, parameters.subject_n,
                                            parameters.hand, mode="train")

    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size,
                             shuffle=True, num_workers=1)

    # local
    train_dataset = torch.utils.data.TensorDataset(torch.randn((20, 1, 204, 250)),
                               torch.ones(20),
                               torch.zeros(20))

    trainloader = DataLoader(train_dataset, batch_size=10,
                             shuffle=True, num_workers=1)

    print("train set {}".format(len(train_dataset)))

    nz = 1000
    netG = Generator(nz=nz, ngf=12, nc=1)
    netD = Discriminator(nc=1, ndf=12)

    netD = netD.to(device)
    netG = netG.to(device)

    print(netG)
    print(netD)

    g_params = 0
    d_param = 0
    print("Generator")
    for name, parameter in netG.named_parameters():
        param = parameter.numel()

        print("param {} : {}".format(name, param if parameter.requires_grad
        else 0))
        g_params += param
    print(f"Generator Trainable Params: {g_params}")

    print("Discriminator")
    for name, parameter in netD.named_parameters():
        param = parameter.numel()
        print("param {} : {}".format(name, param if parameter.requires_grad
        else 0))
        d_param += param
    print(f"Discriminator Trainable Params: {d_param}")

    print(f"Total Trainable Params: {g_params + d_param}")


    if not skip_training:
        print("Begin training....")

        optimizerG = Adam(netG.parameters(), lr=parameters.lr,
                          betas=(0.5, 0.999))
        optimizerD = Adam(netD.parameters(), lr=parameters.lr,
                          betas=(0.5, 0.999))

        start_time = timer.time()

        netG, netD, lossG, lossDF, lossDR, D_fake, D_real = train_gan(netG,
              netD, trainloader, optimizerG, optimizerD, device,
              parameters.epochs, parameters.patience, model_path, nz)

        train_time = timer.time() - start_time
        print("Training done in {:.4f}".format(train_time))

        save_pytorch_model(netG, model_path, "'dcgan_g.pth")
        save_pytorch_model(netD, model_path, "dcgan_d.pth")

    # Evaluation, right now print a rps from data randomly generated.
    print("Evaluation...")
    print("last loss of generator : ", lossG)
    print("last loss of discriminator on _fake : ", lossDF)
    print("last loss of discriminator on _real : ", lossDR)
    print('D_fake', D_fake)
    print('D_reali', D_real)


    netG.eval()
    z_val = torch.randn((4, nz, 1, 1)).to(device)

    _fake_val = netG.forward()

    bands = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 70)]
    bp_train = bandpower_multi(_fake_val, fs=250, bands=bands,
                               nperseg=250 / 2, relative=True)

    epoch = range(_fake_val.shape[0])
    fig, axs = plt.subplots(2, 2, figsize=[12, 6])
    fig.suptitle("RPS_train")
    for e, ax in zip(epoch, axs.ravel()):
        im = ax.pcolormesh(bp_train[e, ...])
        fig.colorbar(im, ax=ax)
        ax.set_ylabel("Channels")
        # ax.set_xlabel("Bands")
        ax.locator_params(axis="y", nbins=5)
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_xticklabels(["\u03B4", "\u03B8", "low-\u03B1",
                            "high-\u03B1", "\u03B2", "low-\u03B3"], )
        # ax.set_title("target: {}".format(sample_y_train[e]))
    # plt.savefig(os.path.join(figure_dir, "RPS_epoch_{}_hand_{}.pdf"
    #                          .format(epoch, "right" if hand == 1 else "left")))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
                        help="Figure data directory (default= MEG\Figures)")
    parser.add_argument('--model_dir', type=str, default='MEG\Models',
                        help="Model data directory (default= MEG\Models\)")

    # subject
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")
    parser.add_argument('--hand', type=int, default='0',
                        help="Patient hands: 0 for sx, 1 for dx (default= 0)")

    # Model Parameters
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch_size_valid', type=int, default=30,
                        metavar='N',
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--batch_size_test', type=int, default=30, metavar='N',
                        help='input batch size for  (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        metavar='lr', help='Learning rate (default: 2e-4),')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='wd', help='Weight dacay (default: 5e-4),')

    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--y_measure', type=str, default="pca",
                        help='Y type reshaping (default: pca)')
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')
    parser.add_argument('--desc', type=str, default="Normal test", metavar='N',
                        help='Experiment description (default: Normal test)')

    args = parser.parse_args()

    main(args)


