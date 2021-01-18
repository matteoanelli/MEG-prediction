#%%
"""
    Main script to generate the Linear Model results on the MEG dataset.
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mne import viz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(1, r'')
from  MEG.Utils.utils import *
from MEG.dl.params import SPoC_params


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    # Generate the parameters class.
    parameters = SPoC_params(subject_n=8,
                             hand=0,
                             duration=1.,
                             overlap=0.8,
                             y_measure=args.y_measure)

    # Generate list of input files
    subj_id = "/sub"+str(args.sub)+"/ball0"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss_trans.fif"]) for i in range(1 if args.sub != 3 else 2, 4)]

    # LOCAL
    # subj_n = 8
    # subj_id = "sub" + str(subj_n) + "\\ball"
    # raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]

    # Import and epoch the MEG data
    X, y_left, y_right = import_MEG_no_bp(raw_fnames,
                                          duration=parameters.duration,
                                          overlap=parameters.overlap,
                                          y_measure=parameters.y_measure,
                                          normalize_input=True)   # concentrate the analysis only on the left hand

    print('X shape {}, y shape {}'.format(X.shape, y_left.shape))

    # Select hand
    if parameters.hand == 0:
        X_train, X_test, y_train, y_test = split_data(X, y_left, 0.3)
    else:
        X_train, X_test, y_train, y_test = split_data(X, y_right, 0.3)

    X_train, X_test = np.reshape(X_train, (X_train.shape[0], -1)), np.reshape(X_test, (X_test.shape[0], -1))

    print("Processing hand {}".format("sx" if parameters.hand == 0 else "dx"))
    print('X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}'.format(X_train.shape, y_train.shape,
                                                                                          X_test.shape, y_test.shape))

    reg = LinearRegression()

    #%%
    # Tune the pipeline
    start = time.time()
    print('Start Fitting model ...')
    reg.fit(X_train, y_train)

    print(f'Training time : {time.time() - start}s ')

    #%%
    # Validate the pipeline
    y_new = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_new)
    rmse = mean_squared_error(y_test, y_new, squared=False)
    mae = mean_absolute_error(y_test, y_new)
    r2 = r2_score(y_test, y_new)
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))
    print("r2 score {}".format(r2))
    #%%
    # Plot the y expected vs y predicted.
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(100)
    ax.plot(times, y_new[100:200], color='b', label='Predicted')
    ax.plot(times, y_test[100:200], color='r', label='True')
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title("SPoC: Sub {}, hand {}, {} prediction".format(str(parameters.subject_n),
                                                         "sx" if parameters.hand == 0 else "dx",
                                                         parameters.y_measure))
    plt.legend()
    viz.tight_layout()
    plt.savefig(os.path.join(figure_path, 'MEG_SPoC_focus.pdf'))
    plt.show()

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_new))
    ax.plot(times, y_new, color="b", label="Predicted")
    ax.plot(times, y_test, color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title("Sub {}, hand {}, {} prediction".format(str(parameters.subject_n),
                                                         "sx" if parameters.hand == 0 else "dx",
                                                         parameters.y_measure))
    plt.legend()
    plt.savefig(os.path.join(figure_path, 'MEG_SPoC.pdf'))
    plt.show()

    # scatterplot y predicted against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax.scatter(np.array(y_test), np.array(y_new), color="b", label="Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    # plt.legend()
    plt.savefig(os.path.join(figure_path, "Scatter.pdf"))
    plt.show()


    # %%
    # Save the model.
    name = 'MEG_SPoC.p'
    save_skl_model(reg, model_path, name)

    # log the model
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('R2', r2)

        mlflow.log_artifact(os.path.join(figure_path, 'MEG_SPoC_focus.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, 'MEG_SPoC.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, 'MEG_SPoC_Components_Analysis.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, "Scatter.pdf"))
        mlflow.sklearn.log_model(reg, "models")

if __name__ == "__main__":
    # main(sys.argv[1:])

    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument('--sub', type=int, default='8',
                        help="Subject number (default= 8)")
    parser.add_argument('--hand', type=int, default='0',
                        help="Patient hands: 0 for sx, 1 for dx (default= 0)")

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\\',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
                        help="Figure data directory (default= MEG\Figures)")
    parser.add_argument('--model_dir', type=str, default='MEG\Models',
                        help="Model data directory (default= MEG\Models\)")

    # Model Parameters
    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.8, metavar='N',
                        help='overlap of time window (default: 0.8s)')
    parser.add_argument('--y_measure', type=str, default="movement",
                        help='Y type reshaping (default: movement)')
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')

    args = parser.parse_args()

    main(args)