#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=30000M
#SBATCH --cpus-per-task=1
#SBATCH --array=2-10
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/swap_out_%A_%a.log
#SBATCH --gres=gpu:1

n=$(($SLURM_ARRAY_TASK_ID + 1))
iteration=`sed -n "${n} p" parameters.csv`

IFS=';' read data figures models sub hand bs bsv bst epochs lr bias duration overlap patience y exp snl skern tnl tkern maxp ffnl ffhc drop act <<< $iteration

echo "data is $data"
echo "figures is $figures"
echo "models is $models"
echo "sub is $sub"
echo "hand is $hand"
echo "bs is $bs"
echo "bsv is $bsv"
echo "bst is $bst"
echo "epochs is $epochs"
echo "lr is $lr"
echo "bias is $bias"
echo "duration is $duration"
echo "overlap is $overlap"
echo "patience is $patience"
echo "y measure is $y"
echo "experiment is $exp"
echo "spatial nl is $snl"
echo "skern is $skern"
echo "temporal nl is $tnl"
echo "tkern is $tkern"
echo "max-pooling is $maxp"
echo "MLP nl is $ffnl"
echo "MLP hidden channels is $ffhc"
echo "dropout is $drop"
echo "activation fun is $act"

mkdir -p tmp/$SLURM_ARRAY_TASK_ID


srun python MEG/dl/DL_MEG.py --data_dir $data --figure_dir $figures --model_dir tmp/$SLURM_ARRAY_TASK_ID --sub $sub --hand $hand --batch_size $bs --batch_size_valid $bsv --batch_size_test $bst --epochs $epochs --learning_rate $lr --duration $duration --overlap $overlap --patience $patience --y_measure $y --experiment $exp --s_n_layer $snl --s_kernel_size $skern --t_n_layer $tnl --t_kernel_size $tkern --max_pooling $maxp --ff_n_layer $ffnl --ff_hidden_channels $ffhc --dropout $drop --activation $a