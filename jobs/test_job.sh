#!/bin/bash
#SBATCH --job-name=test
#SBATCH --error=jobs/%j/output.txt
#SBATCH --output=jobs/%j/output.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:01:00
cd $HOME/repos/PlaNet_Equil_reconstruction
mkdir -p jobs/$SLURM_JOB_ID
echo "running job $SLURM_JOB_ID"
srun python test_job_n.py