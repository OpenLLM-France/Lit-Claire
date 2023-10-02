#!/bin/bash
#SBATCH --job-name=pretrain_lora
#SBATCH --output=pretrain_lora.out
#SBATCH --error=pretrain_lora.out
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
srun python pretrain_lora.py \
--data_dir $SCRATCH/../commun/lit-redpajama-sample \
--checkpoint_dir $WORK/../commun/Claire/checkpoints/tiiuae/falcon-7b \
--precision bf16-true