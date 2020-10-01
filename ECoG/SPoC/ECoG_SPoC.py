#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""

import argparse
import time

import mlflow
import mlflow.sklearn
from mne import viz
from mne.decoding import SPoC
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

from utils import *

sys.path.insert(1, r'')
# from  ECoG.Utils.utils import *
from ECoG.SPoC.utils import standard_scaling
from ECoG.SPoC.SPoC_param import SPoC_params

import matplotlib.pyplot as plt

mne.set_config("MNE_LOGGING_LEVEL", "WARNING")

def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    parameters = SPoC_params(subject_n=args.sub,
                             finger=args.finger,
                             duration=args.duration,
                             overlap=args.overlap)


    #%%

    file_name = "/sub" + str(parameters.subject_n) + "_comp.mat"
    sampling_rate = 1000

    #%%
    # Example
    X, y = import_ECoG(data_dir, file_name, parameters.finger)
    X = filter_data(X, sampling_rate)
    X = standard_scaling(X).squeeze(-1)

    print("Example of fingers position : {}".format(y[0]))
    print("epochs with events generation")
    epochs = create_epoch(X, sampling_rate, duration=parameters.duration, overlap=parameters.overlap, ds_factor=1)

    X = epochs.get_data()

    #%%
    y = y_resampling(y, X.shape[0])

    print(X.shape)
    print(y.shape)

    # %%
    print("X shape {}, y shape {}".format(X.shape, y.shape))
    X_train, X_test, y_train, y_test = split_data(X, y, 0.3)
    print(
        "X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )
    )

    pipeline = Pipeline([('Spoc', SPoC(log=True, reg='oas', rank='full')),
                         ('Ridge', Ridge())])

    # %%
    cv = KFold(n_splits=10, shuffle=False)
    tuned_parameters = [{'Spoc__n_components': list(map(int, list(np.arange(2, 30))))}]

    clf = GridSearchCV(pipeline, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=4, cv=cv, verbose=2
                       )
    # %%
    start = time.time()
    print('Start Fitting model ...')
    clf.fit(X_train, y_train)

    print(f'Training time : {time.time() - start}s ')
    print('Number of cross-validation splits folds/iteration: {}'.format(clf.n_splits_))
    print('Best Score and parameter combination: ')

    print(clf.best_score_)
    print(clf.best_params_['Spoc__n_components'])
    # %%
    y_new = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_new)
    rmse = mean_squared_error(y_test, y_new, squared=False)
    mae = mean_absolute_error(y_test, y_new)
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))
    # %%
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(100)
    ax.plot(times, y_new[100:200], color='b', label='Predicted')
    ax.plot(times, y_test[100:200], color='r', label='True')
    ax.set_xlabel("Times")
    ax.set_ylabel("Finger Movement")
    ax.set_title("Sub {}, finger {} prediction".format(str(parameters.subject_n), parameters.finger))
    plt.legend()
    viz.tight_layout()
    plt.savefig(os.path.join(figure_path, 'ECoG_SPoC_focus.pdf'))
    plt.show()

    # plot y_new against the true value
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(len(y_new))
    ax.plot(times, y_new, color="b", label="Predicted")
    ax.plot(times, y_test, color="r", label="True")
    ax.set_xlabel("Times")
    ax.set_ylabel("Finger Movement")
    ax.set_title("Sub {}, finger {} prediction".format(str(parameters.subject_n), parameters.finger))
    plt.legend()
    plt.savefig(os.path.join(figure_path, 'ECoG_SPoC.pdf'))
    plt.show()

    # %%
    n_components = np.ma.getdata(clf.cv_results_['param_Spoc__n_components'])
    MSEs = clf.cv_results_['mean_test_score']
    # %%

    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax.plot(n_components, MSEs, color='b')
    ax.set_xlabel('Number of SPoC components')
    ax.set_ylabel('MSE')
    ax.set_title('SPoC Components Analysis')
    # plt.legend()
    plt.xticks(n_components, n_components)
    viz.tight_layout()
    plt.savefig(os.path.join(figure_path, 'ECoG_SPoC_Components_Analysis.pdf'))
    plt.show()

    # %%
    name = 'ECoG_SPoC.p'
    save_skl_model(clf, model_path, name)

    # log the model
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)

        mlflow.log_param("n_components", n_components)

        mlflow.log_artifact(os.path.join(figure_path, 'ECoG_SPoC_focus.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, 'ECoG_SPoC.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, 'ECoG_SPoC_Components_Analysis.pdf'))
        mlflow.sklearn.log_model(clf, "models")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # subject
    parser.add_argument('--sub', type=int, default='1',
                        help="Subject number (default= 1)")
    parser.add_argument('--finger', type=int, default='0',
                        help="Finger (default= 0)")

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop\BCICIV_4_mat\\',
                        help="Input data directory (default= Z:\Desktop\BCICIV_4_mat)")
    parser.add_argument('--figure_dir', type=str, default='ECoG\Figures',
                        help="Figure data directory (default= ECoG\Figures)")
    parser.add_argument('--model_dir', type=str, default='ECoG\Models',
                        help="Model data directory (default= ECoG\Models\)")

    # Model Parameters
    parser.add_argument('--duration', type=float, default=1., metavar='N',
                        help='Duration of the time window  (default: 1s)')
    parser.add_argument('--overlap', type=float, default=0.8, metavar='N',
                        help='overlap of time window (default: 0.8s)')
    parser.add_argument('--experiment', type=int, default=0, metavar='N',
                        help='Mlflow experiments id (default: 0)')

    args = parser.parse_args()

    main(args)
