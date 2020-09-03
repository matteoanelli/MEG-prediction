#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""

from utils import *
import sys, getopt

from mne.decoding import SPoC as SPoc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mne
import time

import matplotlib.pyplot as plt

mne.set_config("MNE_LOGGING_LEVEL", "WARNING")

def usage():
    print('\SPoC_MEG.py -i <data_dir> -f <figure_fir> -m <model_dir>')

def main(argv):


    try:
        opts, args = getopt.getopt(argv, "i:f:m:h")

    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    figure_path = "ECoG\Figures"
    model_path = "ECoG\Models"
    data_dir = "C:\\Users\\anellim1\Develop\Thesis\BCICIV_4_mat\\"

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
    #%%
    # data_dir  = os.environ['DATA_PATH']

    file_name = "sub1_comp.mat"
    sampling_rate = 1000

    #%%
    # Example
    X, y = import_ECoG(data_dir, file_name, 0)
    X = filter_data(X, sampling_rate)

    print("Example of fingers position : {}".format(y[0]))
    print("epochs with events generation")
    epochs = create_epoch(X, sampling_rate, duration=1., overlap=0.1, ds_factor=1)


    X = epochs.get_data()

    #%%
    y = y_resampling(y, X.shape[0])

    # %%
    print("X shape {}, y shape {}".format(X.shape, y.shape))
    X_train, X_test, y_train, y_test = split_data(X, y, 0.3)
    print(
        "X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )
    )

    pipeline = Pipeline([("Spoc", SPoc(log=True, reg="oas", rank="full")), ("Ridge", Ridge())])

    # tuned_parameters = [{'Spoc__n_components': list(np.arange(2, 5))}]
    cv = KFold(n_splits=10, shuffle=False)
    tuned_parameters = [{"Spoc__n_components": list(map(int, list(np.arange(2, 16))))}]

    clf = GridSearchCV(pipeline, tuned_parameters, scoring="neg_mean_squared_error", n_jobs=2, cv=cv, verbose=2,)
    #%%
    start = time.time()
    print("Start Fitting model ...")
    clf.fit(X_train, y_train)

    print(f"Training time : {time.time() - start}s ")
    print("Number of cross-validation splits folds/iteration: {}".format(clf.n_splits_))
    print("Best Score and parameter combination: ")

    print(clf.best_score_)
    print(clf.best_params_)

    y_new_train = clf.predict(X_train)
    y_new = clf.predict(X_test)

    print("mean squared error {}".format(mean_squared_error(y_test, y_new)))
    print("mean absolute error {}".format(mean_absolute_error(y_test, y_new)))

    # plot y_new against the true value
    # fig, ax = plt.subplots(1, 2, figsize=[10, 4])
    # times = np.arange(y_new.shape[0])
    # ax[0].plot(times, y_new, color='b', label='Predicted')
    # ax[0].plot(times, y_test, color='r', label='True')
    # ax[0].set_xlabel('Epoch')
    # ax[0].set_ylabel('Finger Movement')
    # ax[0].set_title('SPoC Finger Movement')
    # # times = np.arange(y_new_train.shape[0])
    # # ax[1].plot(times, y_new_train, color='b', label='Predicted')
    # # ax[1].plot(times, y_train, color='r', label='True')
    # # ax[1].set_xlabel('Epoch')
    # # ax[1].set_ylabel('Finger Movement')
    # # ax[1].set_title('SPoC Finger Movement training')
    # # plt.legend()
    # mne.viz.tight_layout()
    # plt.show()
    # %%

    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(y_new.shape[0])
    ax.plot(times, y_new, color="b", label="Predicted")
    ax.plot(times, y_test, color="r", label="True")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Finger Movement")
    ax.set_title("SPoC Finger Movement")
    plt.legend()
    mne.viz.tight_layout()
    plt.savefig(os.path.join(figure_path, "SPoC_Finger_Prediction_half_w0.1.pdf"))
    plt.show()


    # y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    # print(mean_squared_error(y, y_pred))
    # %%
    n_components = np.ma.getdata(clf.cv_results_["param_Spoc__n_components"])
    MSEs = clf.cv_results_["mean_test_score"]
    # %%

    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    ax.plot(n_components, MSEs, color="b")
    ax.set_xlabel("Number of SPoC components")
    ax.set_ylabel("MSE")
    ax.set_title("SPoC Components Analysis")
    # plt.legend()
    plt.xticks(n_components, n_components)
    mne.viz.tight_layout()
    plt.savefig(os.path.join(figure_path, "SPoC_Components_Analysis_half_w0.1.pdf"))
    plt.show()

    # %%
    name = "BaselineModel_SPoC_Best_Filtered_half.p"
    save_skl_model(clf, model_path, name)

if __name__ == "__main__":
    main(sys.argv[1:])
