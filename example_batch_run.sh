#!/bin/bash

#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=32000M
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/out_%A_%a.log
#SBATCH --partition=short-hsw

case $SLURM_ARRAY_TASK_ID in

    1)  sub=9 ;;
    2)  sub=3 ;;
    3)  sub=4 ;; 
    4)  sub=5 ;;
	5)  sub=6 ;;
	6)  sub=7 ;; 
    7)  sub=9 ;;
esac

echo "sub is $sub"

srun python MEG/Dataset/create_preprocessed_dataset.py --data_dir /m/nbe/scratch/strokemotor/healthy_trans/ --out_dir /m/nbe/scratch/strokemotor/healthy_trans/preprocessed/ --sub $sub