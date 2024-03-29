#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=12000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-2
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/Within_RPS_MLP_out_%A_%a.log
#SBATCH --gres=gpu:1
# if resenet add --constraint='pascal|volta'
n=$(($SLURM_ARRAY_TASK_ID + 1))
iteration=`sed -n "${n} p" within_parameters.csv`

IFS=';' read data figures models sub hand bs bsv bst epochs lr wd patience exp desc <<< $iteration

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
echo "wd is $wd"
echo "exp is $exp"
echo "desc is $desc"

mkdir -p tmp/$SLURM_ARRAY_TASK_ID


srun python MEG/dl/within_sub.py --data_dir $data --figure_dir $figures --model_dir tmp/$SLURM_ARRAY_TASK_ID --sub $sub --hand $hand --batch_size $bs --batch_size_valid $bsv --batch_size_test $bst --epochs $epochs --learning_rate $lr --weight_decay $wd  --patience $patience --experiment $exp --desc $desc

# srun python MEG/dl/psd_sub.py --data_dir $data --figure_dir $figures --model_dir tmp/$SLURM_ARRAY_TASK_ID --sub $sub --hand $hand --batch_size $bs --batch_size_valid $bsv --batch_size_test $bst --epochs $epochs --learning_rate $lr --weight_decay $wd  --patience $patience --y_measure $y --experiment $exp --desc $desc
