#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=30000M
#SBATCH --cpus-per-task=2
#SBATCH --array=1-3
#SBATCH --output=/scratch/work/anellim1/MEG-prediction/slurm/ECoG_SPoC__%A_%a.log

n=$((SLURM_ARRAY_TASK_ID+1))
echo "slurm job id is $SLURM_ARRAY_TASK_ID"
iteration=`sed -n "${n} p" ECoG_SPoC_parameters.csv`

IFS=, read data figures models sub finger duration overlap exp <<< $iteration

echo "data is $data"
echo "figures is $figures"
echo "models is $models"
echo "sub is $sub"
echo "finger is $finger"
echo "duration is $duration"
echo "overlap is $overlap"
echo "exp is $exp"

srun python ECoG/SPoC/ECoG_SPoC.py --data_dir $data --figure_dir $figures --model_dir $models --sub $SLURM_ARRAY_TASK_ID --finger $finger --duration $duration --overlap $overlap --experiment $exp
