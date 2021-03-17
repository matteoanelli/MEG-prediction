#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=15000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-8
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/SPoC_out_%A_%a.log

n=$((SLURM_ARRAY_TASK_ID+1))
echo "slurm job id is $SLURM_ARRAY_TASK_ID"
iteration=`sed -n "${n} p" SPoC_parameters.csv`

IFS=, read data figures models sub hand duration overlap y exp alpha <<< $iteration

echo "data is $data"
echo "figures is $figures"
echo "models is $models"
echo "sub is $sub"
echo "hand is $hand"
echo "duration is $duration"
echo "overlap is $overlap"
echo "y is $y"
echo "exp is $exp"
echo "alpha is $alpha"

srun python MEG/SPoC/SPoC_MEG.py --data_dir $data --figure_dir $figures --model_dir $models --sub $sub --hand $hand --duration $duration --overlap $overlap --y_measure $y --experiment $exp
