import getopt
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import argparse

from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split

sys.path.insert(1, r'')

from MEG.dl.train import train
from MEG.dl.MEG_Dataset import MEG_Dataset
from MEG.dl.models import SCNN_swap, DNN, Sample
from MEG.dl.params import Params

# TODO maybe better implementation
from  MEG.Utils.utils import *

def main(args):
    #TODO use arg.parse instead

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir


    subj_id = "/sub"+str(args.sub)+"/ball"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]


    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    parameters = Params(subject_n=args.sub,
                        batch_size=args.batch_size,
                        valid_batch_size=args.batch_size_valid,
                        test_batch_size=args.batch_size_test,
                        epochs=args.epochs,
                        lr=args.learning_rate,
                        duration=args.duration,
                        overlap=args.overlap,
                        patience=args.patience,
                        device=device,
                        y_measure=args.y_measure)

    dataset = MEG_Dataset(raw_fnames, parameters.duration, parameters.overlap, normalize_input=True)

    train_len, valid_len, test_len = len_split(len(dataset))
    train_dataset, valid_test, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=False, num_workers=1)
    validloader = DataLoader(valid_test, batch_size=parameters.valid_batch_size, shuffle=False, num_workers=1)
    testloader = DataLoader(test_dataset, batch_size=parameters.test_batch_size, shuffle=False, num_workers=1)

    # data, _ = iter(trainloader).next()
    # print('trainloader : {}'.format(data))
    #
    # data, _ = iter(testloader).next()
    # print('testloader : {}'.format(data))
    #
    # data, _ = iter(validloader).next()
    # print('validloader : {}'.format(data))


    # net = LeNet5(in_channel=204, n_times=1001)
    net = Sample()
    print(net)

    # Training loop or model loading
    if not skip_training:
        print("Begin training....")

        optimizer = Adam(net.parameters(), lr=parameters.lr)
        loss_function = torch.nn.MSELoss()


        net, train_loss, valid_loss = train(net, trainloader, validloader, optimizer, loss_function,
                                            parameters.device, parameters.epochs, parameters.patience, model_path)




        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
        plt.plot(range(1, len(valid_loss)+1), valid_loss,label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss))+1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5) # consistent scale
        plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        image1 = fig
        plt.savefig(os.path.join(figure_path, "loss_plot.png"))

    if not skip_training:
        # Save the trained model
        save_pytorch_model(net, model_path, "Baselinemodel_SCNN_swap.pth")
    else:
        # Load the model
        net = SCNN_swap()
        net = load_pytorch_model(net, os.path.join(model_path, "Baselinemodel_SCNN_swap_half.pth"), "cpu")


    # Evaluation
    print("Evaluation...")
    net.eval()
    y_pred = []
    y = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(parameters.device), labels.to(parameters.device)
            y.extend(list(labels[:, 0]))
            y_pred.extend((list(net(data))))

    print('SCNN_swap...')
    # Calculate Evaluation measures
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    print("mean squared error {}".format(mse))
    print("mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(100)
    ax.plot(times, y_pred[0:100], color="b", label="Predicted")
    ax.plot(times, y[0:100], color="r", label="True")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acceleration")
    ax.set_title("Accelerometer prediction")
    plt.legend()
    plt.savefig(os.path.join(figure_path, "Accelerometer_prediction_SCNN_swap_half_01_{:.4f}.pdf".format(mse)))
    plt.show()


    # log the model
    with mlflow.start_run() as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)

        mlflow.log_artifact(os.path.join(figure_path, "Accelerometer_prediction_SCNN_swap_half_01_{:.4f}.pdf"
                                         .format(mse)))
        mlflow.log_artifact(os.path.join(figure_path, "loss_plot.png"))
        mlflow.pytorch.log_model(net, model_path)




if __name__ == "__main__":
    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument('--sub', type=int, default='8',
                        help="Input data directory (default= 8)")

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
                        help="Figure data directory (default= MEG\Figures)")
    parser.add_argument('--model_dir', type=str, default='MEG\Models',
                        help="Model data directory (default= MEG\Models\)")

    # Model Parameters
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch_size_valid', type=int, default=30, metavar='N',
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--batch_size_test', type=int, default=30, metavar='N',
                        help='input batch size for  (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=int, default=1e-5, metavar='lr',
                        help='Learning rate (default: 1e-5),')
    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.8, metavar='N',
                        help='overlap of time window (default: 0.8s)')
    parser.add_argument('--patience', type=int, default=20, metavar='N',
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--y_measure', type=str, default="movement",
                        help='Y type reshaping (default: movement)')

    args = parser.parse_args()

    main(args)

# TODO y normalization
# TODO early stopping
# TODO Validation set

