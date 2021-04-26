"""
    Hyper-parameter random search-spcae generation.
"""

import argparse
import random

import pandas as pd


def generate_parameters(param_grid, times, fix, data_dir, figure_dir,
                        model_dir):

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
            sampled_grid["batch_size"] = random.choice(
                param_grid.get("batch_size"))

        if "learning_rate" not in sampled_grid:
            sampled_grid["learning_rate"] = round(
                random.uniform(*param_grid.get("learning_rate")), 5)

        if "weight_decay" not in sampled_grid:
            sampled_grid["weight_decay"] = random.choice(
                param_grid.get("weight_decay"))

        if "s_kernel_size" not in sampled_grid:
            sampled_grid["s_kernel_size"] = random.choice(
                param_grid.get("s_kernel_size")
            )

        if "batch_norm" not in sampled_grid:
            sampled_grid["batch_norm"] = random.choice(
                param_grid.get("batch_norm")
            )

        if "s_drop" not in sampled_grid:
            sampled_grid["s_drop"] = random.choice(
                param_grid.get("s_drop")
            )

        if "mlp_n_layer" not in sampled_grid:
            sampled_grid["mlp_n_layer"] = random.choice(
                param_grid.get("mlp_n_layer")
            )

        if "mlp_hidden" not in sampled_grid:
            sampled_grid["mlp_hidden"] = random.choice(
                param_grid.get("mlp_hidden")
            )

        if "mlp_drop" not in sampled_grid:
            sampled_grid["mlp_drop"] = random.choice(
                param_grid.get("mlp_drop")
            )

        random_grid.append(sampled_grid)
        i += 1

    print("generated {} combination".format(i))

    return random_grid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--data_dir', type=str, default='Z:\Desktop',
                        help="Input data directory (default= Z:\Desktop\\)")
    parser.add_argument('--figure_dir', type=str, default='MEG\Figures',
                        help="Figure data directory (default= MEG\Figures)")
    parser.add_argument('--model_dir', type=str, default='MEG\Models',
                        help="Model data directory (default= MEG\Models\)")

    description = "PSD_cnn_spatial_ADAM_l2_loss_wd_shceduler10_no_max_spatial"

    param_grid = {
        "sub": [1, 2, 3, 5, 6, 7, 8, 9],
        "hand": [0, 1],
        "batch_size": [80, 100, 120],
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [5e-4, 5e-5],
        "s_kernel_size": [
            [204],
            [104, 101],
            [154, 51],
            [104, 51, 51],
        ],
        "batch_norm": [True, False],
        "s_drop": [True, False],
        "mlp_n_layer": [2, 3], # add check if is 1
        "mlp_hidden": [1024, 512, 256, 128], # maybe add 1024
        "mlp_drop": [0.3, 0.4, 0.5]
    }

    args = parser.parse_args()

    fix_param = {
        "batch_size_valid": 30,
        "batch_size_test": 30,
        "hand": 0,
        "sub": 8,
        "epochs": 200,
        "patience": 40,
        "batch_norm": True,
        "s_drop": False,
        "experiment": 49,
        "desc": description,
    }

    random_search = generate_parameters(param_grid, 20, fix_param,
                                        args.data_dir, args.figure_dir,
                                        args.model_dir)

    df = pd.DataFrame(random_search)
    df = df[['data_dir', 'figure_dir', 'model_dir', 'sub', 'hand',
             'batch_size', 'batch_size_valid', 'batch_size_test', "epochs",
             'learning_rate','weight_decay', 'patience', "batch_norm",
             "s_kernel_size", "s_drop", "mlp_n_layer", "mlp_hidden",
             "mlp_drop", 'experiment', "desc"]]

    print(df)
    # np.savetxt("parameters.csv", df, delimiter=";")
    df.to_csv("PSD_parameters.csv", index=False, sep=";")


