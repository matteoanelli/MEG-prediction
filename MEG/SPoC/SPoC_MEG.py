#%%
import time
import sys, getopt

from mne.decoding import SPoC
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from mne import viz

import matplotlib.pyplot as plt
# TODO maybe better implementation
sys.path.insert(1, r'')
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
    subj_id = "sub"+str(subj_n)+"\\ball"
    raw_fnames = ["".join([data_dir, subj_id, str(i), "_sss.fif"]) for i in range(1, 4)]

    X, y, _ = import_MEG(raw_fnames, duration=1., overlap=0.)   # concentrate the analysis only on the left hand

    print('X shape {}, y shape {}'.format(X.shape, y.shape))
    X_train, X_test, y_train, y_test = split_data(X, y, 0.3)
    print('X_train shape {}, y_train shape {} \n X_test shape {}, y_test shape {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    pipeline = Pipeline([('Spoc', SPoC(log=True, reg='oas', rank='full')),
                         ('Ridge', Ridge())])

    # %%
    cv = KFold(n_splits=10, shuffle=False)
    tuned_parameters = [{'Spoc__n_components': list(map(int, list(np.arange(2, 30))))}]

    clf = GridSearchCV(pipeline, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=4, cv=cv, verbose=2)
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
    y_new_train = clf.predict(X_train)
    y_new = clf.predict(X_test)
    MSE = mean_squared_error(y_test, y_new)
    print('mean squared error {}'.format(MSE))
    print('mean absolute error {}'.format(mean_absolute_error(y_test, y_new)))
    #%%
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    times = np.arange(100)
    ax.plot(times, y_new[100:200], color='b', label='Predicted')
    ax.plot(times, y_test[100:200], color='r', label='True')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hand movement')
    ax.set_title('SPoC hand Movement')
    plt.legend()
    viz.tight_layout()
    plt.savefig(os.path.join(figure_path, 'MEG_SPoC_MSE_{:.4f}.pdf'.format(mean_squared_error(y_test, y_new))))
    plt.show()


    # y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    # print(mean_squared_error(y, y_pred))
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
    plt.savefig(os.path.join(figure_path, 'MEG_SPoC_Components_Analysis_MSE_{:.4f}.pdf'.format(MSE)))
    plt.show()

    # %%
    name = 'MEG_SPoC_MSE_{:.4f}.p'.format(MSE)
    save_skl_model(clf, model_path, name)

if __name__ == "__main__":
    main(sys.argv[1:])