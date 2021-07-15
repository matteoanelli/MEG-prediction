#%%
"""
    Main script to generate the SVM baseline model on MEG dataset.
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mne import viz
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(1, r"")
from MEG.Utils.utils import *
from MEG.dl.params import SPoC_params
from sklearn.svm import SVR


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    # Generate the parameters class.
    parameters = SPoC_params(
        subject_n=args.sub,
        hand=args.hand,
        duration=args.duration,
        overlap=args.overlap,
        y_measure=args.y_measure,
        alpha=0,
    )

    # Import and epoch the MEG data
    _, y_train, rps_train = import_MEG_within_subject_ivan(
        data_dir, args.sub, args.hand, "train"
    )

    _, y_val, rps_val = import_MEG_within_subject_ivan(
        data_dir, args.sub, args.hand, "val"
    )

    _, y_test, rps_test = import_MEG_within_subject_ivan(
        data_dir, args.sub, args.hand, "test"
    )

    # Select hand

    rps_train = rps_train.reshape(rps_train.shape[0], -1)
    rps_val = rps_val.reshape(rps_val.shape[0], -1)
    rps_test = rps_test.reshape(rps_test.shape[0], -1)

    rps_train = np.array(rps_train).astype(np.float64)
    y_train = np.array(y_train[..., args.hand].squeeze()).astype(np.float64)

    rps_val = np.array(rps_val).astype(np.float64)
    y_val = np.array(y_val[..., args.hand].squeeze()).astype(np.float64)

    rps_test = np.array(rps_test).astype(np.float64)
    y_test = np.array(y_test[..., args.hand].squeeze()).astype(np.float64)

    print("Processing hand {}".format("sx" if parameters.hand == 0 else "dx"))

    start = time.time()
    print("Start Fitting model ...")

    best_pipeline = None
    best_mse_valid = np.Inf
    best_rmse_valid = 0
    best_r2_valid = 0
    best_alpha = 0
    best_n_comp = 0
    # for _C in [2, 10, 30, 50, 100]:
    for _C in [2, 10, 30, 50, 100]:
        print("Processing C = ", _C)
        regressor = SVR(kernel='rbf', C=_C)

        regressor.fit(rps_train, y_train)

        # Validate the pipeline
        print("evaluation parameters C :{}".format(_C))

        y_new = regressor.predict(rps_val)
        mse = mean_squared_error(y_val, y_new)
        rmse = mean_squared_error(y_val, y_new, squared=False)
        # mae = mean_absolute_error(y_test, y_new)
        r2 = r2_score(y_val, y_new)

        if mse < best_mse_valid:
            print("saving new best model mse {} ---> {}".format(best_mse_valid,
                                                                mse))

            best_regressor = regressor
            best_mse_valid = mse
            best_r2_valid = r2
            best_rmse_valid = rmse
            best_C = _C


    print(f"Training time : {time.time() - start}s ")
    print("Best parameter C: ", best_C)

    print("Validation set")
    print("mean squared error valid {}".format(best_mse_valid))
    print("root mean squared error valid{}".format(best_rmse_valid))
    print("r2 score {}".format(best_r2_valid))

    #%%
    # Test the pipeline
    print("Test set")
    y_new = best_regressor.predict(rps_test)
    mse = mean_squared_error(y_test, y_new)
    rmse = mean_squared_error(y_test, y_new, squared=False)
    # mae = mean_absolute_error(y_test, y_new)
    r2 = r2_score(y_test, y_new)
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    # print("mean absolute error {}".format(mae))
    print("r2 score {}".format(r2))

    #%%
    # Plot the y expected vs y predicted.
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(100)
    ax.plot(times, y_new[100:200], color="b", label="Predicted")
    ax.plot(times, y_test[100:200], color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title(
        "SVM: Sub {}, hand {}, {} prediction".format(
            str(parameters.subject_n),
            "sx" if parameters.hand == 0 else "dx",
            parameters.y_measure,
        )
    )
    plt.legend()
    viz.tight_layout()
    plt.savefig(os.path.join(figure_path, "MEG_SVM_focus.pdf"))
    plt.show()

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_new))
    ax.plot(times, y_new, color="b", label="Predicted")
    ax.plot(times, y_test, color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title(
        "Sub {}, hand {}, {} prediction".format(
            str(parameters.subject_n),
            "sx" if parameters.hand == 0 else "dx",
            parameters.y_measure,
        )
    )
    plt.legend()
    plt.savefig(os.path.join(figure_path, "MEG_SVM.pdf"))
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
    name = "MEG_SVR.p"
    save_skl_model(best_pipeline, model_path, name)

    # log the model
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        # mlflow.log_metric('MAE', mae)
        mlflow.log_metric("R2", r2)

        mlflow.log_metric("RMSE_v", best_rmse_valid)
        mlflow.log_metric("R2_v", best_r2_valid)

        mlflow.log_param("C", best_C)
       
        mlflow.log_artifact(os.path.join(figure_path, "MEG_SVM_focus.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "MEG_SVM.pdf"))
        mlflow.sklearn.log_model(best_pipeline, "models")


if __name__ == "__main__":
    # main(sys.argv[1:])

    # main(sys.argv[1:])

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument(
        "--sub", type=int, default="8", help="Subject number (default= 8)"
    )
    parser.add_argument(
        "--hand",
        type=int,
        default="0",
        help="Patient hands: 0 for sx, 1 for dx (default= 0)",
    )

    # Directories
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Z:\Desktop\\",
        help="Input data directory (default= Z:\Desktop\\)",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="MEG\Figures",
        help="Figure data directory (default= MEG\Figures)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="MEG\Models",
        help="Model data directory (default= MEG\Models\)",
    )

    # Model Parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        metavar="N",
        help="Duration of the time window  (default: 1s)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        metavar="N",
        help="overlap of time window (default: 0.8s)",
    )
    parser.add_argument(
        "--y_measure",
        type=str,
        default="movement",
        help="Y type reshaping (default: movement)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        metavar="N",
        help="Mlflow experiments id (default: 0)",
    )

    args = parser.parse_args()

    main(args)
