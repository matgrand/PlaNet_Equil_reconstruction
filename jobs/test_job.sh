#!/bin/bash
#SBATCH --job-name=test
#SBATCH --error=jobs/output.%j.txt
#SBATCH --output=jobs/output.%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:01:00
cd $HOME/repos/PlaNet_Equil_reconstruction
mkdir -p jobs/%j
echo "running job %j"
srun python test_job_n.py -c=config/config_train_mg.yml -n=%j