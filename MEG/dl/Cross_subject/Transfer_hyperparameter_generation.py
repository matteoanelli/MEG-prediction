
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

        if "batch_size_test" not in sampled_grid:
            sampled_grid["batch_size_test"] = random.choice(param_grid.get("batch_size_test"))

        if "learning_rate" not in sampled_grid:
            sampled_grid["learning_rate"] = round(random.uniform(*param_grid.get("learning_rate")), 5)

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

    description = "Transfer_learning_exp_fine_tune_MLP_shuffle"

    param_grid = {
        "sub": [1, 2, 3, 5, 6, 7, 8, 9],
        "hand": [0, 1],
        "batch_size_test": [60, 80, 100],
        "learning_rate": [1e-2, 1e-5],
    }

    args = parser.parse_args()

    fix_param = {
        "hand": 0,
        "sub": 8,
        "epochs": 100,
        "patience": 20,
        "y_measure": "pca",
        "experiment": 34,
        "run": "05cec4f7f84e48feafeded2a737540d3",
        "desc": description,
    }
    random_search = generate_parameters(param_grid, 30, fix_param, args.data_dir, args.figure_dir, args.model_dir)

    df = pd.DataFrame(random_search)
    df = df[['data_dir', 'figure_dir', 'model_dir', 'sub', 'hand', 'batch_size_test', 'learning_rate', 'experiment',
             "run", "desc"]]

    print(df)
    # np.savetxt("parameters.csv", df, delimiter=";")
    df.to_csv("trans_parameters.csv", index=False, sep=";")
