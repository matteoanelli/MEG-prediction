#!/bin/bash

#SBATCH --time=00:25:00
#SBATCH --mem-per-cpu=5000M
#SBATCH --cpus-per-task=4
#SBATCH --array=8
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/PSD_out_%A_%a.log
#SBATCH --partition=short-hsw

case $SLURM_ARRAY_TASK_ID in

    1)  sub=1 ;;
    2)  sub=2 ;;
    3)  sub=3 ;;
    4)  sub=4 ;;
    5)  sub=5 ;;
    6)  sub=6 ;;
    7)  sub=7 ;;
    8)  sub=8 ;;
    9)  sub=9 ;;
esac

echo "sub is $sub"

# srun python MEG/Dataset/create_preprocessed_dataset.py --data_dir /m/nbe/scratch/strokemotor/healthy_trans/ --out_dir /m/nbe/scratch/strokemotor/healthy_trans/preprocessed/ --sub $sub
# srun python MEG/bp_gen.py --data_dir /m/nbe/scratch/strokemotor/healthy_trans/preprocessed/ --sub $sub
#  srun python MEG/Dataset/plot_y_distr.py --data_dir /m/nbe/scratch/strokemotor/healthy_trans/preprocessed/ --figure_dir /scratch/work/anellim1/Figures --sub $sub --hand 1
# srun python MEG/Dataset/PSD_plot.py --data_dir /m/nbe/scratch/strokemotor/healthy_trans/ --figure_dir /scratch/work/anellim1/Figures --sub $sub
srun python MEG/Dataset/welch_gen.py --data_dir /m/nbe/scratch/strokemotor/healthy_trans/preprocessed/ --sub $sub
