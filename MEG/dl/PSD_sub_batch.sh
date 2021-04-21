#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=12000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1-5
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/PSD_cnn_out_%A_%a.log
#SBATCH --gres=gpu:1
# if resenet add --constraint='pascal|volta'
n=$(($SLURM_ARRAY_TASK_ID + 1))
iteration=`sed -n "${n} p" PSD_parameters.csv`

IFS=';' read data figures models sub hand bs bsv bst epochs lr wd patience batch_norm s_kernel_size s_drop mlp_n_layer mlp_hidden mlp_drop exp desc <<< $iteration

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
echo "batch norm is $batch_norm"
echo "s_kernel_size is $s_kernel_size"
echo "s_drop is $s_drop"
echo "mlp_n_layer is $mlp_n_layer"
echo "mlp_hidden is $mlp_hidden"
echo "mlp_drop is $mlp_drop"
echo "exp is $exp"
echo "desc is $desc"

mkdir -p tmp/$SLURM_ARRAY_TASK_ID

srun python MEG/dl/psd_sub.py --data_dir $data --figure_dir $figures --model_dir tmp/$SLURM_ARRAY_TASK_ID --sub $sub --hand $hand --batch_size $bs --batch_size_valid $bsv --batch_size_test $bst --epochs $epochs --learning_rate $lr --weight_decay $wd  --patience $patience --batch_norm $batch_norm --s_kernel_size $s_kernel_size --s_drop $s_drop --mlp_n_layer $mlp_n_layer --mlp_hidden $mlp_hidden --mlp_drop $mlp_drop --experiment $exp --desc $desc
