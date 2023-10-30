# Claire

This is the code repository used to train Claire on the supercomputer [Jean Zay](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html).

Claire is a reasonably sized LLM specialized for French conversational data
(typically, transcribed and diarized spontaneous oral speech).

* [Setup](#setup)
* [Finetune a model](#finetune-a-model)
* [Check the model and make it available](#check-the-model-and-make-it-available)
* [Acknowledgements](#acknowledgements)

## Setup

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

Those steps are only needed if you want to start from a Hugging Face model.
The last one download additional files that will be needed for the packaging of the model, at last.
```bash
# MODEL=mistralai/Mistral-7B-v0.1
MODEL=tiiuae/falcon-7b

python lit_gpt/scripts/download.py --repo_id $MODEL
python lit_gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL
python download_config.py --repo_id $MODEL --checkpoint_dir checkpoints/$MODEL
```

On Jean Zay, you can do that from the folder `$WORK/../commun/Claire`,
so that all foundation models can be found in `$WORK/../commun/Claire/checkpoints`.

### Prepare data

```bash
MODEL=tiiuae/falcon-7b
FOUNDATION_MODEL_DIR=$WORK/../commun/Claire/checkpoints/$MODEL
DATA_DIR=$SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded_8_grouped/$MODEL

python prepare_data.py \
    --source_path       $WORK/../commun/Claire/data_raw/full \
    --checkpoint_dir    $FOUNDATION_MODEL_DIR \
    --destination_path  $DATA_DIR
```

### Launch training

An example command to launch pre-training on 8 GPU:
```bash
MODEL=tiiuae/falcon-7b
DATA_DIR=$SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded_8_grouped/$MODEL
FOUNDATION_MODEL_DIR=$WORK/../commun/Claire/checkpoints/$MODEL
TRAINING_DIR=$SCRATCH/../commun/Claire/pretrain-Claire-7B-v0.06_mono/lora/$MODEL

python pretrain.py \
--data_dir       $DATA_DIR \
--checkpoint_dir $FOUNDATION_MODEL_DIR \
--out_dir        $TRAINING_DIR \
--devices 8 \
--language fr \
--precision bf16-true \
--micro_batch_size 12 \
--batch_size 132 \
--num_epochs 1000 \
--max_checkpoints 20 \
--save_interval 1800 \
--eval_interval 1800 \
--enable_validation true \
--early_stopping 2 \

```


#### On Jean-Zay

```bash
sbatch slurm/pretrain_<<version>>.slurm
```
In the `.slurm` file, these 3 arguments should be equal:
- `#SBATCH --gres=gpu:8`
- `#SBATCH --ntasks-per-node=8`
- `srun python pretrain.py --devices 8`
  
These 2 arguments should be equal:
- `#SBATCH --nodes=1`
- `srun python pretrain.py --num_nodes 1`

Training checkpoints and monitoring log can be found under `out_dir`, standard output and error should be recorded in that same folder.

On Jean Zay, you can check the status of the job with `squeue -u $USER`
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
100813    gpu_p5 pretrain  ugs29jo  R       0:52      1 jean-zay-iam36
```
cancel the job with `scancel 100813`, connect to the node with `ssh jean-zay-iam36` (on which you can run `nvidia-smi`)

### Offline validation

```bash
MODEL=tiiuae/falcon-7b
DATA_DIR=$SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded_8_grouped/$MODEL
TRAINING_DIR=$SCRATCH/../commun/Claire/pretrain/pretrain-Claire-7B-v0.0.1/lora/$MODEL

srun --ntasks=1 --gres=gpu:1 -C a100 --qos=qos_gpu-dev \
--output=$OUTDIR/validation.out --error=$OUTDIR/validation.out \
--time=00:10:00 \
python validate_pretrain.py \
--data_dir       $DATA_DIR \
--out_dir        $TRAINING_DIR \
--language fr \
--max 40 --batch_size 8 \
--precision bf16-true
```

## Check the model and make it available

### Quick test the model

If trained with LoRA, you can first merge the weights, with a command like:
```bash
MODEL=tiiuae/falcon-7b
TRAINING_DIR=$WORK/../commun/Claire/pretrain/lora/$MODEL
TRAINED_MODEL_PATH=$TRAINING_DIR/iter-020883-ckpt.pth  # lit_model_lora_finetuned.pth
FOUNDATION_MODEL_DIR=$WORK/../commun/Claire/checkpoints/$MODEL
SAVE_DIR=$WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-v0.0.1

srun --ntasks=1 --gres=gpu:1 --constraint=a100 \
python utils/merge_lora.py \
    --lora_path       $TRAINED_MODEL_PATH \
    --checkpoint_dir  $FOUNDATION_MODEL_DIR \
    --save_path       $SAVE_DIR/lit_model.pth \
```
This generates the merged model `lit_model.pth` in the specified `$SAVE_DIR`.

You can then test the model with a single prompt:
```bash
srun --ntasks=1 --gres=gpu:1 --constraint=a100 --qos=qos_gpu-dev \
python lit_gpt/generate/base.py \
    --prompt "[Intervenant 1:] Bonjour, mon nom est" \
    --checkpoint_dir $SAVE_DIR
```
or test it interactively:
```bash
srun --ntasks=1 --gres=gpu:1 --constraint=a100 --qos=qos_gpu-dev --pty \
python lit_gpt/chat/base.py \
    --checkpoint_dir $SAVE_DIR
```

Note: you can also test LoRA weights directly, without merging first, by using `lora.py` instead of `base.py` in the two commands above.

### Convert trained Lit-GPT model to transformers and upload it to Hugging Face

You can convert the model to transormers with the following command.
Use option `--repo_id` if and only if you want to upload the model.
```bash
MODEL=tiiuae/falcon-7b
TRAINING_DIR=$WORK/../commun/Claire/pretrain/lora/$MODEL
TRAINED_MODEL_PATH=$TRAINING_DIR/iter-020883-ckpt.pth
SAVE_DIR=$WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-v0.0.1

python convert_litgpt_to_transformers.py \
    --input_path $TRAINED_MODEL_PATH \
    --output_dir $SAVE_DIR \
    --repo_id    OpenLLM-France/Claire-7B-v0.0.1
```
You will need to provide your [User Access Tokens](https://huggingface.co/settings/tokens).

The steps done by this script are:
* Copy relevant files from the foundation model checkpoint folder
  (This folder should be in `$SAVE_DIR/hparams.json`, and can also be specified with option `--checkpoint_dir`)
* If needed, merge LoRA weights
* Convert the model in [lit-gpt](https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/convert_lit_checkpoint.py) format (`lit_model.pth`) to a model in the [transformers](https://github.com/huggingface/transformers) format (`pytorch_model.bin`).
* If needed, split the big model into chunks of <10 GB (ex: `pytorch_model-00001-of-00002.bin`, `pytorch_model-00002-of-00002.bin`, `pytorch_model.bin.index.json`)
* If asked (with `--repo_id`):
  * Create the Hugging Face repo if it does not exist
  * Upload the model and its companion files

### Update Hugging Face model card

The model card [README.md](hf_files/v00/README.md),
as well as files requided to make an endpoint ([handler.py](hf_files/v00/handler.py) and [requirements.txt](hf_files/v00/requirements.txt))
can be updated on the Hugging Face model hub page [OpenLLM-France/Claire-7B-v0.0.1](https://huggingface.co/OpenLLM-France/Claire-7B-v0.0.1) with the following command:
```bash
python utils/hf_upload_model.py \
OpenLLM-France/Claire-7B-v0.0.1 \
--message "<<your commit message>>"
```

The model weights can be updated directly with `convert_litgpt_to_transformers.py` or with
```bash
python utils/hf_upload_model.py \
OpenLLM-France/Claire-7B-v0.0.1 \
--input_dir $SAVE_DIR \
--message "Upload weights"
```

You will need to provide your [User Access Tokens](https://huggingface.co/settings/tokens).

## Acknowledgements

* [Lightning](https://github.com/Lightning-AI/lightning):  Deep Learning framework to train and deploy neural networks.
* [Lit-GPT](https://github.com/Lightning-AI/lit-gpt): Hackable implementation of state-of-the-art open-source Large Language Models.
* [HuggingFace](https://huggingface.co/models): Model hub containing state-of-the-art open-source Large Language Models.

This work was granted access to the HPC resources of IDRIS under the allocation 20XX-AD011014561 made by GENCI
