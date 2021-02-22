
"""
    Hyper-parameter random search-spcae generation.
"""

import argparse
import random

import pandas as pd


def generate_parameters(param_grid, times, fix, data_dir, figure_dir, model_dir):

    random_grid = []
    i = 0
    while i < times:
        sampled_grid = {
            "data_dir": data_dir,
            "figure_dir": figure_dir,
            "model_dir": model_dir,
        }
        if fix is not None:
            sampled_grid.update(fix)

        if "sub" not in sampled_grid:
            sampled_grid["sub"] = random.choice(param_grid.get("sub"))

        if "hand" not in sampled_grid:
            sampled_grid["hand"] = random.choice(param_grid.get("hand"))

        if "batch_size" not in sampled_grid:
            sampled_grid["batch_size"] = random.choice(param_grid.get("batch_size"))

        if "learning_rate" not in sampled_grid:
            sampled_grid["learning_rate"] = round(random.uniform(*param_grid.get("learning_rate")), 5)

        if "weight_decay" not in sampled_grid:
            sampled_grid["weight_decay"] = random.choice(param_grid.get("weight_decay"))



        random_grid.append(sampled_grid)
        i += 1

    print("generated {} combination".format(i))

    return random_grid

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
                        help="Figure data directory (default= MEG\Figures)")
    parser.add_argument('--model_dir', type=str, default='MEG\Models',
                        help="Model data directory (default= MEG\Models\)")

    description = "y_pca_trans_rps_mnet_exp_SGD_batchnorm_new_arch"

    param_grid = {
        "sub": [1, 2, 3, 5, 6, 7, 8, 9],
        "hand": [0, 1],
        "batch_size": [80, 100, 120],
        "learning_rate": [3e-3, 1e-5],
        "y_measure": ["pca", "left_single_1"],
        "weight_decay": [5e-3, 5e-4, 5e-5]
    }

    args = parser.parse_args()

    fix_param = {
        "batch_size_valid": 30,
        "batch_size_test": 30,
        "hand": 0,
        "sub": 8,
        "epochs": 100,
        "patience": 20,
        "y_measure": "pca",
        "experiment": 30,
        "desc": description
    }
    random_search = generate_parameters(param_grid, 20, fix_param, args.data_dir, args.figure_dir, args.model_dir)

    df = pd.DataFrame(random_search)
    df = df[['data_dir', 'figure_dir', 'model_dir', 'sub', 'hand', 'batch_size', 'batch_size_valid',
             'batch_size_test', "epochs", 'learning_rate','weight_decay', 'patience', 'y_measure', 'experiment', "desc"]]

    print(df)
    # np.savetxt("parameters.csv", df, delimiter=";")
    df.to_csv("cross_parameters.csv", index=False, sep=";")
