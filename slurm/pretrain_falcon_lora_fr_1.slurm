#!/bin/bash
#SBATCH --job-name=pretrain_lora
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=192:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=8
#SBATCH -C a100

## load environment
module purge
module load cpuarch/amd
module load anaconda-py3/2023.03
conda activate claire

## launch script on every node
set -x

MODEL=tiiuae/falcon-7b_bos
OUTDIR=$SCRATCH/../commun/Claire/pretrain-Claire-7B-v0.02/lora/$MODEL

# execute script
srun --output=$OUTDIR/pretrain_lora.out --error=$OUTDIR/pretrain_lora.out \
python pretrain_lora.py \
--devices 2 \
--num_nodes 1 \
--data_dir $SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded/$MODEL \
--checkpoint_dir $WORK/../commun/Claire/checkpoints/$MODEL \
--language fr \
--out_dir $OUTDIR \
--precision bf16-true \
--enable_validation false \
