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

### Download then convert Hugging Face model to Lit-GPT format

```bash
# MODEL=mistralai/Mistral-7B-v0.1
MODEL=tiiuae/falcon-7b

python lit_gpt/scripts/download.py --repo_id $MODEL
python lit_gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL
```

On Jean Zay, you can do that from the folder `$WORK/../commun/Claire`.

### prepare data
```
python prepare_data.py \
    --source_path       $WORK/../commun/Claire/data_raw/full \
    --checkpoint_dir    $WORK/../commun/Claire/checkpoints/tiiuae/falcon-7b \
    --destination_path  $SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded/tiiuae/falcon-7b
```

### launch training
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


### merge lora
```
MODEL=tiiuae/falcon-7b
srun --ntasks=1 --gres=gpu:1 --constraint=a100 \
python merge_lora.py \
    --checkpoint_dir $WORK/../commun/Claire/checkpoints/$MODEL \
    --lora_dir       $WORK/../commun/Claire/pretrain/lora/$MODEL \
    --lora_pth_name  lit_model_lora_finetuned.pth \
    --save_path      $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B/lit_model.pth \
    --precision      bf16-true
```
The merged model `lit_model.pth` can be found under `save_path`

copy the *.json files from Falcon-7b to Claire-7B, which are required for the configuration and tokenizer information.
```
cp $WORK/../commun/Claire/checkpoints/tiiuae/falcon-7b/*.json \
    $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B/
```

### test the merged model

test the model with a single prompt
```
srun --ntasks=1 --gres=gpu:1 --constraint=a100 \
python lit_gpt/generate/base.py \
    --prompt "Hello, my name is" \
    --checkpoint_dir $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B
```

test the model interactively
```
srun --ntasks=1 --gres=gpu:1 --constraint=a100 --pty \
python lit_gpt/chat/base.py \
    --checkpoint_dir $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B
```

### convert Lit-GPT model to Hugging Face format
```
python lit_gpt/scripts/convert_lit_checkpoint.py \
    --checkpoint_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B/lit_model.pth \
    --output_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B/pytorch_model.bin \
    --config_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B/lit_config.json
```

### split pytorch_model.bin into smaller shards
```
python download_config.py \
    --folder_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B
```
```
srun --ntasks=1 --gres=gpu:1 --constraint=a100 --cpus-per-task=8 \
python split_model.py \
    --folder_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B
```

### upload the converted model to Hugging Face
```
python upload_model.py \
    --folder_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B \
    --repo_id OpenLLM-France/Claire-7B \
    --create_repo true
```
`--create_repo true`: create a new Hugging Face repo with `repo_id`, then upload files.  
`--create_repo false`: upload files to the existing Hugging Face repo with `repo_id`  
You'll need to provide your [User Access Tokens](https://huggingface.co/settings/tokens).
