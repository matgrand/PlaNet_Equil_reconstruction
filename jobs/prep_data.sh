#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --error=jobs/%j.txt
#SBATCH --output=jobs/%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=01:59:00
cd $HOME/repos/PlaNet_Equil_reconstruction
srun git pull && jupyter nbconvert prepare_dataset.ipynb --to python && python prepare_dataset.py && rm -rf prepare_dataset.py