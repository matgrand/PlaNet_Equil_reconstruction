#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --error=jobs/output.%j.txt
#SBATCH --output=jobs/output.%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=00:59:00
cd $HOME/repos/PlaNet_Equil_reconstruction
srun jupyter nbconvert prepare_dataset.ipynb --to python && python prepare_dataset.py && rm -rf prepare_dataset.py
