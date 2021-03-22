#%%
"""
    Main script to generate the SPoC results on the MEG dataset.
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mne import viz
from mne.decoding import SPoC
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(1, r"")
from MEG.Utils.utils import *
from MEG.dl.params import SPoC_params


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    file_name = "data.hdf5"

    # Generate the parameters class.
    parameters = SPoC_params(
        subject_n=args.sub,
        hand=args.hand,
        duration=args.duration,
        overlap=args.overlap,
        y_measure=args.y_measure,
        alpha=args.alpha,
    )

    X_train, y_train, _ = import_MEG_cross_subject_train(
        data_dir, file_name, parameters.subject_n, parameters.hand
    )

    X_test, y_test, _ = import_MEG_cross_subject_test(
        data_dir, file_name, parameters.subject_n, parameters.hand
    )

    # Required conversion and double float precision.

    if parameters.hand == 0:
        X_train, y_train = (
            np.array(X_train.squeeze()).astype(np.float64),
            np.array(y_train[..., 0].squeeze()).astype(np.float64),
        )
        X_test, y_test = (
            np.array(X_test.squeeze()).astype(np.float64),
            np.array(y_test[..., 0].squeeze()).astype(np.float64),
        )
    else:
        X_train, y_train = (
            np.array(X_train.squeeze()).astype(np.float64),
            np.array(y_train[..., 1].squeeze()).astype(np.float64),
        )
        X_test, y_test = (
            np.array(X_test.squeeze()).astype(np.float64),
            np.array(y_test[..., 1].squeeze()).astype(np.float64),
        )

    # Add the transfer part to the train_set
    test_len, transfer_len = len_split_cross(X_test.shape[0])
    X_transfer = X_test[:transfer_len, ...]
    X_test = X_test[transfer_len:, ...]
    X_train = np.concatenate((X_train, X_transfer), axis=0)

    y_transfer = y_test[:transfer_len, ...]
    y_test = y_test[transfer_len:, ...]
    y_train = np.concatenate((y_train, y_transfer), axis=0)

    print("Processing hand {}".format("sx" if parameters.hand == 0 else "dx"))
    print(
        "X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )
    )

    pipeline = Pipeline(
        [
            ("Spoc", SPoC(log=True, reg="oas", rank="full")),
            ("Ridge", Ridge(alpha=parameters.alpha)),
        ]
    )

    # %%
    # Initialize the cross-validation pipeline and grid search
    cv = KFold(n_splits=5, shuffle=False)
    tuned_parameters = [
        {"Spoc__n_components": list(map(int, list(np.arange(1, 30, 5))))}
    ]

    clf = GridSearchCV(
        pipeline,
        tuned_parameters,
        scoring=["neg_mean_squared_error", "r2"],
        n_jobs=-1,
        cv=cv,
        refit="neg_mean_squared_error",
        verbose=3,
    )

    #%%
    # Tune the pipeline
    start = time.time()
    print("Start Fitting model ...")
    clf.fit(X_train, y_train)
    print(clf)

    print(f"Training time : {time.time() - start}s ")
    print(
        "Number of cross-validation splits folds/iteration: {}".format(
            clf.n_splits_
        )
    )
    print("Best Score and parameter combination: ")

    print(clf.best_score_)
    print(clf.best_params_["Spoc__n_components"])
    print("CV results")
    print(clf.cv_results_)
    print("Number of splits")
    print(clf.n_splits_)
    #%%
    # Validate the pipeline
    y_new = clf.predict(X_test)
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
    ax.plot(times, y_new[100:200], color="b", label="Predicted")
    ax.plot(times, y_test[100:200], color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("{}".format(parameters.y_measure))
    ax.set_title(
        "SPoC: Sub {}, hand {}, {} prediction".format(
            str(parameters.subject_n),
            "sx" if parameters.hand == 0 else "dx",
            parameters.y_measure,
        )
    )
    plt.legend()
    viz.tight_layout()
    plt.savefig(os.path.join(figure_path, "MEG_SPoC_focus.pdf"))
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
    plt.savefig(os.path.join(figure_path, "MEG_SPoC.pdf"))
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
    n_components = np.ma.getdata(clf.cv_results_["param_Spoc__n_components"])
    MSE_valid = clf.cv_results_["mean_test_neg_mean_squared_error"][0]
    R2_valid = clf.cv_results_["mean_test_r2"][0]
    # %%
    # Save the model.
    name = "MEG_SPoC.p"
    save_skl_model(clf, model_path, name)

    # log the model
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.log_metric("RMSE_Valid", MSE_valid)
        mlflow.log_metric("R2_Valid", R2_valid)

        mlflow.log_param(
            "n_components", clf.best_params_["Spoc__n_components"]
        )
        mlflow.log_param("alpha", parameters.alpha)

        mlflow.log_artifact(os.path.join(figure_path, "MEG_SPoC_focus.pdf"))
        mlflow.log_artifact(os.path.join(figure_path, "MEG_SPoC.pdf"))
        mlflow.log_artifact(
            os.path.join(figure_path, "MEG_SPoC_Components_Analysis.pdf")
        )
        mlflow.sklearn.log_model(clf, "models")


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
        default="pca",
        help="Y type reshaping (default: pca)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        metavar="N",
        help="Mlflow experiments id (default: 0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2,
        metavar="N",
        help="Ridge alpha value (default: 2)",
    )

    args = parser.parse_args()

    main(args)
