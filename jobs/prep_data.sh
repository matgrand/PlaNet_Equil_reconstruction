#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --error=output.%j.err
#SBATCH --output=output.%j.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=00:22:00
cd $HOME/repos/PlaNet_Equil_reconstruction
echo "Starting job"
srun jupyter nbconvert prepare_dataset.ipynb --to python && python prepare_dataset.py && rm -rf prepare_dataset.py
echo "Job finished"