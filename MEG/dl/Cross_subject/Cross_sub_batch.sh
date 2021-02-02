#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=30000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/RPS_MLP_out_%A_%a.log
#SBATCH --gres=gpu:1
# if resenet add --constraint='pascal|volta'
n=$(($SLURM_ARRAY_TASK_ID + 1))
iteration=`sed -n "${n} p" cross_parameters.csv`

IFS=';' read data figures models sub hand bs bsv bst epochs lr patience y exp  <<< $iteration

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
echo "exp is $exp"

mkdir -p tmp/$SLURM_ARRAY_TASK_ID


srun python MEG/dl/Cross_subject/Cross_DL_MEG.py --data_dir $data --figure_dir $figures --model_dir tmp/$SLURM_ARRAY_TASK_ID --sub $sub --hand $hand --batch_size $bs --batch_size_valid $bsv --batch_size_test $bst --epochs $epochs --learning_rate $lr  --patience $patience --y_measure $y --experiment $exp

