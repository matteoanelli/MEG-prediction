#%%
import argparse
import sys
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mne import viz
from mne.decoding import SPoC
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(1, r'')
from  MEG.Utils.utils import *
from MEG.dl.params import SPoC_params


def main(args):

    data_dir = args.data_dir
    figure_path = args.figure_dir
    model_path = args.model_dir

    parameters = SPoC_params(subject_n=args.sub,
                             hand=args.hand,
                             duration=args.duration,
                             overlap=args.overlap,
                             y_measure=args.y_measure)

    subj_id = "/sub" + str(parameters.subject_n) + "/ball"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1 if args.sub != 3 else 2, 4)]
    # raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 2)]

    X, y, _ = import_MEG(raw_fnames, duration=parameters.duration, overlap=parameters.overlap,
                         y_measure=parameters.y_measure, normalize_input=True)   # concentrate the analysis only on the left hand

    print('X shape {}, y shape {}'.format(X.shape, y.shape))

    X_train, X_test, y_train, y_test = split_data(X, y, 0.3)
    print('X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}'.format(X_train.shape, y_train.shape,
                                                                                          X_test.shape, y_test.shape))

    pipeline = Pipeline([('Spoc', SPoC(log=True, reg='oas', rank='full')),
                         ('Ridge', Ridge())])

    # %%
    cv = KFold(n_splits=10, shuffle=False)
    tuned_parameters = [{'Spoc__n_components': list(map(int, list(np.arange(2, 30))))}]

    clf = GridSearchCV(pipeline, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=4, cv=cv, verbose=2
                       )
    #%%
    start = time.time()
    print('Start Fitting model ...')
    clf.fit(X_train, y_train)

    print(f'Training time : {time.time() - start}s ')
    print('Number of cross-validation splits folds/iteration: {}'.format(clf.n_splits_))
    print('Best Score and parameter combination: ')

    print(clf.best_score_)
    print(clf.best_params_['Spoc__n_components'])
    #%%
    y_new = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_new)
    rmse = mean_squared_error(y_test, y_new, squared=False)
    mae = mean_absolute_error(y_test, y_new)
    print("mean squared error {}".format(mse))
    print("root mean squared error {}".format(rmse))
    print("mean absolute error {}".format(mae))
    #%%
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
    plt.savefig(os.path.join(figure_path, 'MEG_SPoC_Components_Analysis.pdf'))
    plt.show()

    # %%
    name = 'MEG_SPoC.p'
    save_skl_model(clf, model_path, name)

    # log the model
    with mlflow.start_run(experiment_id=args.experiment) as run:
        for key, value in vars(parameters).items():
            mlflow.log_param(key, value)

        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)

        mlflow.log_param("n_components", n_components)

        mlflow.log_artifact(os.path.join(figure_path, 'MEG_SPoC_focus.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, 'MEG_SPoC.pdf'))
        mlflow.log_artifact(os.path.join(figure_path, 'MEG_SPoC_Components_Analysis.pdf'))
        mlflow.sklearn.log_model(clf, "models")

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