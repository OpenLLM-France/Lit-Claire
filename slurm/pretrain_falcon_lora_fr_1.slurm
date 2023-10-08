#!/bin/bash
#SBATCH --job-name=pretrain_lora
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=8
#SBATCH -C a100

## load environment
module purge
module load cpuarch/amd
module load anaconda-py3/2023.03
conda activate claire

## launch script on every node
set -x

MODEL=tiiuae/falcon-7b
OUTDIR=$SCRATCH/../commun/Claire/pretrain-Claire-7B-v0.02/lora/$MODEL
mkdir -p $OUTDIR

# execute script
srun --output=$OUTDIR/pretrain_lora.out --error=$OUTDIR/pretrain_lora.out \
python pretrain_lora.py \
--devices 8 \
--num_nodes 1 \
--data_dir $SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded_8_grouped/$MODEL \
--checkpoint_dir $WORK/../commun/Claire/checkpoints/$MODEL \
--language fr \
--out_dir $OUTDIR \
--precision bf16-true \
--enable_validation false \
--num_epochs 1000 \
--max_checkpoints 20 \
