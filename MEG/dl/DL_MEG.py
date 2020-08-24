import getopt
import sys

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split

sys.path.insert(1, r'')

from MEG.dl.train import train
from MEG.dl.MEG_Dataset import MEG_Dataset
from MEG.dl.models import SCNN_swap, DNN, LeNet5, DNN_seq, LeNet5_seq, SCNN_swap_seq

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
    duration = 1.
    overlap = 0.

    # Set skip_training to False if the model has to be trained, to True if the model has to be loaded.
    skip_training = False

    # Set the torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device = {}".format(device))

    dataset = MEG_Dataset(raw_fnames, duration, overlap, normalize_input=True)

    train_dataset, valid_test, test_dataset = random_split(dataset,
                                                           [
                                                               round(len(dataset)*0.7),
                                                               round(len(dataset)*0.15),
                                                               round(len(dataset)*0.15)
                                                           ])

    trainloader = DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=1)
    validloader = DataLoader(valid_test, batch_size=50, shuffle=False, num_workers=1)

    testloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    # net = LeNet5_seq(in_channel=204, n_times=1001)
    net = SCNN_swap_seq()
    print(net)

    # Training loop or model loading
    if not skip_training:
        print("Begin training...")
        EPOCHS = 500
        optimizer = Adam(net.parameters(), lr=0.00001)
        loss_function = torch.nn.MSELoss()
        patient = 20

        train(net, trainloader, validloader, optimizer, loss_function, device, EPOCHS, patient)




    if not skip_training:
        # Save the trained model
        save_pytorch_model(net, model_path, "Baselinemodel_SCNN_swap.pth")
    else:
        # Load the model
        net = SCNN_swap()
        net = load_pytorch_model(net, os.path.join(model_path, "Baselinemodel_SCNN_swap.pth"), "cpu")

    # Evaluation
    print("Evaluation...")
    net.eval()
    y_pred = []
    y = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            y.extend(list(labels[:, 0]))
            y_pred.extend((list(net(data))))


    print('SCNN_swap_seq...')
    # Calculate Evaluation measures
    mse = mean_squared_error(y, y_pred)
    print("mean squared error {}".format(mse))
    print("mean absolute error {}".format(mean_absolute_error(y, y_pred)))

    print(y_pred[:10])

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(100)
    ax.plot(times, y_pred[100:200], color="b", label="Predicted")
    ax.plot(times, y[100:200], color="r", label="True")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acceleration")
    ax.set_title("Accelerometer prediction")
    plt.legend()
    plt.savefig(os.path.join(figure_path, "Accelerometer_prediction_SCNN_swap_seq{:.4f}.pdf".format(mse)))
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])

# TODO y normalization
# TODO early stopping
# TODO Validation set

