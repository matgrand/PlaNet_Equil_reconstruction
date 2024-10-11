#!/bin/bash
#SBATCH --job-name=train
#SBATCH --error=jobs/%j.txt
#SBATCH --output=jobs/%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a40:1
cd $HOME/repos/PlaNet_Equil_reconstruction
echo "running job $SLURM_JOB_ID"
srun jupyter nbconvert mg_train.ipynb --to python && python mg_train.py && rm -rf mg_train.py