#!/bin/bash
#SBATCH --job-name=Mistral-Apache
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=07:00:00
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

MODEL=mistralai/Mistral-7B-v0.1
OUTDIR=$WORK/../commun/Claire/pretrain/Claire-Mistral-7B-Apache-0.1
mkdir -p $OUTDIR

# execute script
srun --output=$OUTDIR/training_log.out --error=$OUTDIR/training_log.out \
python pretrain.py \
--devices 8 \
--num_nodes 1 \
--data_dir $SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded_8_grouped_open/$MODEL \
--checkpoint_dir $WORK/../commun/Claire/checkpoints/$MODEL \
--language fr \
--out_dir $OUTDIR \
--precision bf16-true \
--num_epochs 1000 \
--max_checkpoints 13 \
--enable_validation true \
--early_stopping 2 \
--lora_r 16 \
--lora_alpha 32 \
--batch_size 16 \
--micro_batch_size 8 \
--save_interval 1800 \
--eval_interval 1800
