import argparse
import random

import pandas as pd


def test_parameter(params):

    n_times = params["duration"] * 1000 + 1

    temporal_kernel_size = params["t_kernel_size"]
    temporal_n_block = len(temporal_kernel_size)
    max_pool = params["max_pooling"]
    # max_pool = 2

    n_times_ = n_times
    for i in range(temporal_n_block):
        n_times_ = int((n_times_ - ((temporal_kernel_size[i] - 1) * 2)))
        n_times_ = int(n_times_ / (max_pool if max_pool is not None else 1))

    if n_times_ < 1:
        print(" The reduction factor must be < than n_times. Got reduction to {}"
                         " recalculatig parameters...".format(n_times_))
        return False
    else:
        return True

def generate_parameters(param_grid, times, fix, data_dir, figure_dir, model_dir):

    random_grid = []
    count = 0
    i = 0
    while count < times:
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

        if "duration_overlap" not in sampled_grid:
            sampled_grid["duration_overlap"] = random.choice(param_grid.get("duration_overlap"))
        sampled_grid["duration"] = sampled_grid["duration_overlap"][0]
        sampled_grid["overlap"] = sampled_grid["duration_overlap"][1]
        sampled_grid.pop("duration_overlap")

        if "s_kernel_size" not in sampled_grid:
            sampled_grid["s_kernel_size"] = random.choice(param_grid.get("s_kernel_size"))
        sampled_grid["s_n_layer"] = len(sampled_grid["s_kernel_size"])

        if "t_kernel_size" not in sampled_grid:
            sampled_grid["t_kernel_size"] = random.choice(param_grid.get("t_kernel_size"))
        sampled_grid["t_n_layer"] = len(sampled_grid["t_kernel_size"])

        if "ff_n_layer" not in sampled_grid:
            sampled_grid["ff_n_layer"] = random.choice(param_grid.get("ff_n_layer"))

        if "ff_hidden_channels" not in sampled_grid:
            sampled_grid["ff_hidden_channels"] = random.choice(param_grid.get("ff_hidden_channels"))

        if "dropout" not in sampled_grid:
            sampled_grid["dropout"] = random.choice(param_grid.get("dropout"))

        if "activation" not in sampled_grid:
            sampled_grid["activation"] = random.choice(param_grid.get("activation"))

        if test_parameter(sampled_grid):
            random_grid.append(sampled_grid)
            count += 1
        i += 1

    print("generated {} combination in {} trials".format(count, i))

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

    param_grid = {
        "sub": [8],
        "hand": [0, 1],
        "batch_size": [80, 100, 120],
        "learning_rate": [3e-3, 1e-4],
        "duration_overlap": [(1., 0.8), (1.2, 1.), (1.4, 1.2)],
        "s_kernel_size": [[204], [54, 51, 51, 51], [104, 101], [154, 51], [104, 51, 51]],
        "t_kernel_size": [[20, 10, 10, 8, 5], [16, 8, 5, 5], [10, 10, 10, 10], [100, 75], [250]],
        "ff_n_layer": [2, 3, 4],
        "ff_hidden_channels": [1024, 516, 248],
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "activation": ["relu", "selu", "elu"]
    }

    args = parser.parse_args()

    fix_param = {
        "batch_size_valid": 30,
        "batch_size_test": 30,
        "epochs": 100,
        "bias": False,
        "patience": 20,
        "y_measure": "movement",
        "max_pooling": 2,
        "experiment": 4,
    }
    random_search = generate_parameters(param_grid, 20, fix_param, args.data_dir, args.figure_dir, args.model_dir)

    df = pd.DataFrame(random_search)
    df = df[['data_dir', 'figure_dir', 'model_dir', 'sub', 'hand', 'batch_size', 'batch_size_valid',
             'batch_size_test', "epochs", 'learning_rate', 'bias', 'duration', 'overlap', 'patience', 'y_measure',
             'experiment',
             's_n_layer', 's_kernel_size', 't_n_layer', 't_kernel_size', 'max_pooling',
             'ff_n_layer', 'ff_hidden_channels', 'dropout', 'activation']]

    print(df)
    # np.savetxt("parameters.csv", df, delimiter=";")
    df.to_csv("parameters.csv", index=False, sep=";")
