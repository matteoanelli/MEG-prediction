#!/bin/bash

#SBATCH --time=03:30:00
#SBATCH --mem-per-cpu=25000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-4
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/out_%j.log
#SBATCH --gres=gpu:1

n=$(($SLURM_ARRAY_TASK_ID + 1))
iteration=`sed -n "${n} p" parameters.csv`

IFS=, read data figures models sub hand bs bsv bst epochs lr duration overlap patience y exp <<< $iteration

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
echo "duration is $duration"
echo "overlap is $overlap"
echo "patience is $patience"
echo "y is $y"
echo "exp is $exp"

srun python MEG/dl/DL_MEG.py --data_dir $data --figure_dir $figures --model_dir $models --sub $sub --hand $hand --batch_size $bs --batch_size_valid $bsv --batch_size_test $bst --epochs $epochs --learning_rate $lr --duration $duration --overlap $overlap --patience $patience --y_measure $y --experiment $exp
