#!/bin/bash
#SBATCH --job-name=train
#SBATCH --error=jobs/%j/output.txt
#SBATCH --output=jobs/%j/output.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a40:1
cd $HOME/repos/PlaNet_Equil_reconstruction
mkdir -p jobs/$SLURM_JOB_ID
echo "running job $SLURM_JOB_ID"
srun git pull && jupyter nbconvert mg_train.ipynb --to python && python mg_train.py && rm -rf mg_train.py