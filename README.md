# CLAIRE

## Installation

### Clone the repo
```
git clone --recurse-submodules https://github.com/OpenLLM-France/Claire
```

### Create environment

First create a virtual environment.
Example on Jean Zay:
```bash
module load cpuarch/amd
module load anaconda-py3/2023.03

conda create -y -n claire python=3.10
conda activate claire
```
Example on your own machine:
```bash
python3.10 -m venv env
source env/bin/activate
```

### Install dependencies

Then, install the dependencies (you may want to use `--user` if you don't use a virtual env):
```bash
pip install --no-cache-dir -r requirements.txt
```

## Finetune a model

### Download then convert Hugging Face model to Lit-GPT format

```bash
# MODEL=mistralai/Mistral-7B-v0.1
MODEL=tiiuae/falcon-7b

python lit_gpt/scripts/download.py --repo_id $MODEL
python lit_gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL
```

On Jean Zay, you can do that from the folder `$WORK/../commun/Claire`.

### Prepare data
```
python prepare_data.py \
    --source_path       $WORK/../commun/Claire/data_raw/full \
    --checkpoint_dir    $WORK/../commun/Claire/checkpoints/tiiuae/falcon-7b \
    --destination_path  $SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded/tiiuae/falcon-7b
```

### Launch training
```
sbatch pretrain_lora.slurm
```
These 3 arguments should be equal:
- `#SBATCH --gres=gpu:2`
- `#SBATCH --ntasks-per-node=2`
- `srun python pretrain.py --devices 2`
  
These 2 arguments should be equal:
- `#SBATCH --nodes=1`
- `srun python pretrain.py --num_nodes 1`

training checkpoints and monitoring log can be found under `out_dir`, standard output and error are recorded in `pretrain_lora.out`

on Jean Zay, you can check the status of the job with `squeue -u $USER`
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
100813    gpu_p5 pretrain  ugs29jo  R       0:52      1 jean-zay-iam36
```
cancel the job with `scancel 100681`, connect to the node with `ssh jean-zay-iam36` (on which you can run `nvidia-smi`)

## Check the model and make it available

### Quick test the model

If trained with LoRA, you can first merge the weights, with a command like:
```
MODEL=tiiuae/falcon-7b
srun --ntasks=1 --gres=gpu:1 --constraint=a100 \
python utils/merge_lora.py \
    --lora_path       $WORK/../commun/Claire/pretrain/lora/$MODEL/lit_model_lora_finetuned.pth \
    --checkpoint_dir  $WORK/../commun/Claire/checkpoints/$MODEL \
    --save_path       $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B/lit_model.pth \
```
The merged model `lit_model.pth` can be found under `save_path`

You can then test the model with a single prompt:
```
srun --ntasks=1 --gres=gpu:1 --constraint=a100 --qos=qos_gpu-dev \
python lit_gpt/generate/base.py \
    --prompt "[Intervenant 1:] Bonjour, mon nom est" \
    --checkpoint_dir $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-v0.0.1
```
or test it interactively:
```
srun --ntasks=1 --gres=gpu:1 --constraint=a100 --qos=qos_gpu-dev --pty \
python lit_gpt/chat/base.py \
    --checkpoint_dir $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-v0.0.1
```

Note: you can also test LoRA weights directly, without merging first, by using `lora.py` instead of `base.py` in the two commands above.

### Convert trained Lit-GPT model to transformers and upload it to Hugging Face

You can convert the model to transormers with the following command.
Use option `--repo_id` if and only if you want to upload the model.
```
python convert_litgpt_to_transformers.py \
    --input_path $WORK/../commun/Claire/pretrain-Claire-7B-v0.06_mono/lora/tiiuae/falcon-7b/iter-020883-ckpt.pth \
    --output_dir $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-v0.0.1 \
    --repo_id OpenLLM-France/Claire-7B-v0.0.1
```
You'll need to provide your [User Access Tokens](https://huggingface.co/settings/tokens).
