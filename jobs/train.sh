#!/bin/bash
#SBATCH --job-name=train
#SBATCH --error=output.%j.txt
#SBATCH --output=output.%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a40:1
cd $HOME/repos/PlaNet_Equil_reconstruction
srun jupyter nbconvert mg_train.ipynb.ipynb --to python && python mg_train.ipynb.py && rm -rf mg_train.ipynb.py
