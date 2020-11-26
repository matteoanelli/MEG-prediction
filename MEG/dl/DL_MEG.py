import getopt
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import argparse
import time as timer
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, random_split

sys.path.insert(1, r'')

from MEG.dl.train import train
from MEG.dl.MEG_Dataset import MEG_Dataset
from MEG.dl.models import SCNN_swap, DNN, Sample, SCNN_tunable, LeNet5, ResNet, RPS_MNet_2
from MEG.dl.params import Params_tunable

# TODO maybe better implementation
from  MEG.Utils.utils import *

mne.set_config("MNE_LOGGING_LEVEL", "WARNING")

def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir


    subj_id = "/sub"+str(args.sub)+"/ball0"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss_trans.fif"]) for i in range(1 if args.sub != 3 else 2, 4)]
    # raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]


    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))



    parameters = Params_tunable(subject_n=args.sub,
                                hand=args.hand,
                                batch_size=args.batch_size,
                                valid_batch_size=args.batch_size_valid,
                                test_batch_size=args.batch_size_test,
                                epochs=args.epochs,
                                lr=args.learning_rate,
                                duration=args.duration,
                                overlap=args.overlap,
                                patience=args.patience,
                                device=device,
                                y_measure=args.y_measure,
                                s_n_layer=args.s_n_layer,
                                # s_kernel_size=args.s_kernel_size,
                                s_kernel_size=json.loads(' '.join(args.s_kernel_size)),
                                t_n_layer=args.t_n_layer,
                                # t_kernel_size=args.t_kernel_size,
                                t_kernel_size=json.loads(' '.join(args.t_kernel_size)),
                                max_pooling=args.max_pooling,
                                ff_n_layer=args.ff_n_layer,
                                ff_hidden_channels=args.ff_hidden_channels,
                                dropout=args.dropout,
                                activation=args.activation
                                )

    dataset = MEG_Dataset(raw_fnames,
                          parameters.duration,
                          parameters.overlap,
                          parameters.y_measure,
                          normalize_input=True)

    train_len, valid_len, test_len = len_split(len(dataset))
    print('{} + {} + {} = {}?'.format(train_len, valid_len, test_len, len(dataset)))

    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len], generator = torch.Generator().manual_seed(42))
    # train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True, num_workers=1)
    validloader = DataLoader(valid_test, batch_size=parameters.valid_batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False, num_workers=1)

    # data, _ = iter(trainloader).nexcd t()
    # print('trainloader : {}'.format(data))
    #
    # data, _ = iter(testloader).next()
    # print('testloader : {}'.format(data))
    #
    # data, _ = iter(validloader).next()
    # print('validloader : {}'.format(data))
    with torch.no_grad():
        # Change if RPS itegration
        x, _, _ = iter(trainloader).next()
        # x, _ = iter(trainloader).next()
    n_times = x.shape[-1]

    # Initialize network
    # net = LeNet5(n_times)
    # net = ResNet([2, 2, 2], 64, n_times)
    # net = SCNN_swap(n_times)
    # net = SCNN_tunable(parameters.s_n_layer,
    #                    parameters.s_kernel_size,
    #                    parameters.t_n_layer,
    #                    parameters.t_kernel_size,
    #                    n_times,
    #                    parameters.ff_n_layer,
    #                    parameters.ff_hidden_channels,
    #                    parameters.dropout,
    #                    parameters.max_pooling,
    #                    parameters.activation)
    net = RPS_MNet_2(n_times)

    print(net)
    # Training loop or model loading
    if not skip_training:
        print("Begin training....")

        # Check the optimizer before running (different from model to model)
        optimizer = Adam(net.parameters(), lr=parameters.lr, weight_decay=5e-4)
        # optimizer = SGD(net.parameters(), lr=parameters.lr, weight_decay=5e-4)

        loss_function = torch.nn.MSELoss()
        start_time = timer.time()
        net, train_loss, valid_loss = train(net, trainloader, validloader, optimizer, loss_function,
                                            parameters.device, parameters.epochs, parameters.patience,
                                            parameters.hand, model_path)

        train_time = timer.time() - start_time
        print("Training done in {:.4f}".format(train_time))


        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
        plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss))+1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        # plt.ylim(0, 0.5) # consistent scale
        # plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        image1 = fig
        plt.savefig(os.path.join(figure_path, "loss_plot.pdf"))

    if not skip_training:
        # Save the trained model
        save_pytorch_model(net, model_path, "Baselinemodel_SCNN_swap.pth")
    else:
        # Load the model
        net = SCNN_tunable()
        net = load_pytorch_model(net, os.path.join(model_path, "Baselinemodel_SCNN_swap.pth"), "cpu")


    # Evaluation
    print("Evaluation...")
    net.eval()
    y_pred = []
    y = []
    with torch.no_grad():
        for data, labels, bp in testloader:
            data, labels, bp = data.to(parameters.device), labels.to(parameters.device), bp.to(device)
            y.extend(list(labels[:, parameters.hand, :]))
            y_pred.extend((list(net(data, bp))))

    y = torch.stack(y).to('cpu')
    y_pred = torch.stack(y_pred).to('cpu')

    print('SCNN_swap...')
    # Calculate Evaluation measures
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))

    # plot y_new against the true value focus on 100 timepoints
    # fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    # times = np.arange(100)
    # ax.plot(times, y_pred[0:100], color="b", label="Predicted")
    # ax.plot(times, y[0:100], color="r", label="True")
    # ax.set_xlabel("Times")
    # ax.set_ylabel("{}".format(parameters.y_measure))
    # ax.set_title("Sub {}, hand {}, {} prediction".format(str(parameters.subject_n),
    #                                                      "sx" if parameters.hand == 0 else "dx",
    #                                                      parameters.y_measure))
    # plt.legend()
    # plt.savefig(os.path.join(figure_path, "Times_prediction_focus.pdf"))
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=[10, 4])

    limit = 100
    times = np.arange(limit)
    ax[0].plot(times, y_pred[:limit, 0], color = "b", label = "Predicted")
    ax[0].plot(times, y[:limit, 0], color="r", label="True")
    ax[0].set_xlabel("Times")
    ax[0].set_ylabel("RMSE")
    ax[0].set_title("Sub {}, hand {} x prediction".format(str(parameters.subject_n),
                                                          "sx" if parameters.hand == 0 else "dx"))
    ax[0].legend()

    ax[1].plot(times, y_pred[:limit, 1], color="b", label="Predicted")
    ax[1].plot(times, y[:limit, 1], color="r", label="True")
    ax[1].set_xlabel("Times")
    ax[1].set_title("Sub {}, hand {} y prediction".format(str(parameters.subject_n),
                                                          "sx" if parameters.hand == 0 else "dx"))
    ax[1].legend()

    plt.savefig(os.path.join(figure_path, "Times_prediction_focus.pdf"))
    plt.show()

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_pred))
    ax.plot(times, y_pred, color="b", label="Predicted")
    ax.plot(times, y, color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title("Sub {}, hand {}, {} prediction".format(str(parameters.subject_n),
                                                         "sx" if parameters.hand == 0 else "dx",
                                                         parameters.y_measure))
    plt.legend()
    plt.savefig(os.path.join(figure_path, "Times_prediction.pdf"))
    plt.show()


    # log the model
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_param("Time", train_time)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)

        mlflow.log_artifact(os.path.join(figure_path, "Times_prediction.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "Times_prediction_focus.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "loss_plot.pdf"))
        mlflow.pytorch.log_model(net, "models")




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
    parser.add_argument('--batch_size_valid', type=int, default=30, metavar='N',
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--batch_size_test', type=int, default=30, metavar='N',
                        help='input batch size for  (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='lr',
                        help='Learning rate (default: 1e-3),')
    parser.add_argument('--bias', type=bool, default=False, metavar='N',
                        help='Convolutional layers with bias(default: False)')


    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.8, metavar='N',
                        help='overlap of time window (default: 0.8s)')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--y_measure', type=str, default="movement",
                        help='Y type reshaping (default: movement)')
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')

    # Model architecture parameters
    # Spatial sub-net
    parser.add_argument('--s_n_layer', type=int, default=2, metavar='N',
                        help='Spatial sub-net number of layer (default: 2)')
    parser.add_argument('--s_kernel_size', type=str, default=[104, 101], metavar='N', nargs='+',
                        help='Spatial sub-net kernel sizes (default: [104, 101])')
    # Temporal sub-net
    parser.add_argument('--t_n_layer', type=int, default=5, metavar='N',
                        help='Temporal sub-net number of layer (default: 5)')
    parser.add_argument('--t_kernel_size', type=str, default=[20, 10, 10, 8, 5], metavar='N', nargs='+',
                        help='Spatial sub-net kernel sizes (default: [20, 10, 10, 8, 5])')
    parser.add_argument('--max_pooling', type=int, default=2, metavar='lr',
                        help='Spatial sub-net max-pooling (default: 2)')

    # MLP
    parser.add_argument('--ff_n_layer', type=int, default=3, metavar='N',
                        help='MLP sub-net number of layer (default: 3)')
    parser.add_argument('--ff_hidden_channels', type=int, default=1024, metavar='N',
                        help='MLP sub-net number of hidden channels (default: 1024)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='d',
                        help='MLP dropout (default: 0.5),')

    # Activation
    parser.add_argument('--activation', type=str, default="relu", metavar='N',
                        help='Activation function ti apply (default: relu)')



    args = parser.parse_args()

    main(args)
