import getopt
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

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

def usage():
    print('\SPoC_MEG.py -i <data_dir> -f <figure_fir> -m <model_dir>')

def main(argv):
    #TODO use arg.parse instead
    try:
        opts, args = getopt.getopt(argv, "i:f:m:h")

    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    data_dir = 'Z:\Desktop\\'
    figure_path = 'MEG\Figures'
    model_path = 'MEG\Models'

    for opt, arg in opts:
        if opt == '-h':
            print('\SPoC_MEG.py -i <data_dir> -f <figure_fir> -m <model_dir>')
            sys.exit()
        elif opt in "-i":
            data_dir = arg
        elif opt in "-f":
            figure_path = arg
        elif opt in "-m":
            model_path = arg

    subj_n = 8
    subj_id = "/sub"+str(subj_n)+"/ball"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]


    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    parameters = Params(batch_size=100, valid_batch_size=50, test_batch_size=10, epochs=2,
                        lr=0.00001, duration=1., overlap=0., patient=20, device=device)


    dataset = MEG_Dataset(raw_fnames, parameters.duration, parameters.overlap, normalize_input=True)

    print(len(dataset))
    print('{} {} {}'.format(round(len(dataset)*0.7), round(len(dataset)*0.15-1), round(len(dataset)*0.15)))
    # TODO make it general
    train_dataset, valid_test, test_dataset = random_split(dataset,
                                                           [
                                                               round(len(dataset)*0.7),
                                                               round(len(dataset)*0.15+1),
                                                               round(len(dataset)*0.15)
                                                           ])

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
    # net = net.to(device)

    # Training loop or model loading
    if not skip_training:
        print("Begin training....")
        optimizer = Adam(net.parameters(), lr=parameters.lr)
        loss_function = torch.nn.MSELoss()


        net, train_loss, valid_loss = train(net, trainloader, validloader, optimizer, loss_function,
                                            parameters.device, parameters.epochs, parameters.patient)




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
            data, labels = data.to(device), labels.to(device)
            y.extend(list(labels[:, 0]))
            # print(net(data))
            y_pred.extend((list(net(data))))

    print('SCNN_swap...')
    # Calculate Evaluation measures
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    print("mean squared error {}".format(mse))
    print("mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))

    print(y_pred[:10])

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
    main(sys.argv[1:])

# TODO y normalization
# TODO early stopping
# TODO Validation set

