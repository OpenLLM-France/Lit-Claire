#!/bin/bash
#SBATCH --job-name=training_claire
#SBATCH --output=training_claire.out
#SBATCH --error=training_claire.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=4
#SBATCH -C a100

## load environment
module purge
module load cpuarch/amd
module load anaconda-py3/2023.03
conda activate claire

## launch script on every node
set -x

# execute script
srun python full.py