#!/bin/bash
#SBATCH --job-name=train
#SBATCH --error=jobs/%j/output.txt
#SBATCH --output=jobs/%j/output.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:01:00
cd $HOME/repos/PlaNet_Equil_reconstruction
echo "running job %j"
srun python test_job_n.py -c=config/config_train_mg.yml -n=%j