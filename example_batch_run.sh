#!/bin/bash

#SBATCH --time=03:30:00
#SBATCH --mem-per-cpu=25000M
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/out_%j.log
#SBATCH --gres=gpu:1

case $SLURM_ARRAY_TASK_ID in

    0)  LR=2e-3 ;;
    1)  LR=3e-3 ;;
    2)  LR=4e-3 ;;
    3)  LR=5e-3 ;; 
    4)  LR=8e-4 ;;
esac

srun python MEG/dl/DL_MEG.py --data_dir /scratch/nbe/strokemotor/healthysubjects --figure_dir /scratch/work/anellim1/Figures --model_dir /scratch/work/anellim1/Models --learning_rate $LR --experiment 1 --patience 10

