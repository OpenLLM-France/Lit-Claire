# Lit-Claire

This is the code repository used to train [Claire models](https://huggingface.co/OpenLLM-France/Claire-7B-0.1)
using [⚡ Lightning Fabric](https://lightning.ai/docs/fabric/stable/),
with hints to run on a supercomputer like [Jean Zay](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html).

Claire is a suite of reasonably sized LLM specialized for conversational data
(typically, transcribed and diarized spontaneous oral speech).

* [Setup](#setup)
* [Continual pretraining](#continual-pretraining)
  * [Download and convert foundation model to Lit-GPT format](#download-and-convert-foundation-model-to-lit-gpt-format)
  * [Download raw data](#download-raw-data)
  * [Prepare data](#prepare-data)
  * [Launch training](#launch-training)
  * [Monitoring](#monitoring)
    * [Convergence curves](#convergence-curves)
    * [Offline validation](#offline-validation)
* [Check the model and make it available](#check-the-model-and-make-it-available)
  * [Merge LoRA weights](#merge-lora-weights)
  * [Quick test of the model](#quick-test-of-the-model)
  * [Convert trained Lit-GPT model and upload it to 🤗 Hugging Face](#convert-trained-lit-gpt-model-and-upload-it-to--hugging-face)
     * [Update Hugging Face model card](#update-hugging-face-model-card)
  * [Quantize the model (GGUF format)](#quantize-the-model-gguf-format)
* [Acknowledgements](#acknowledgements)

## Setup

### Clone the repo
```
git clone --recurse-submodules https://github.com/OpenLLM-France/Claire
```

### Create environment

First create a virtual environment.

For example:
```bash
python3.10 -m venv env
source env/bin/activate
```

Or on Jean Zay:
```bash
module load cpuarch/amd
module load anaconda-py3/2023.03

conda create -y -n claire python=3.10
conda activate claire
```

### Install dependencies

Then, install the dependencies (you may want to use `--user` if you don't use a virtual env):
```bash
pip install --no-cache-dir -r requirements.txt
```

You may also want to update torch to the latest version:
```bash
pip install pytorch-lightning==2.1.2
pip install torch==2.2.0.dev20231127+cu121 -f https://download.pytorch.org/whl/nightly/torch/
```

## Continual pretraining

In the following, example bash commands are given for foundation model `tiiuae/falcon-7b`.
Other foundation models can be used, such as `mistralai/Mistral-7B-v0.1`.

### Download and convert foundation model to Lit-GPT format

Those steps are only needed if you want to start from a Hugging Face model.
The last one download additional files that will be needed for the packaging of the trained model in the end.
```bash
MODEL=tiiuae/falcon-7b

python lit_gpt/scripts/download.py --repo_id $MODEL
python lit_gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL
python download_config.py --repo_id $MODEL --checkpoint_dir checkpoints/$MODEL
```

On Jean Zay, you can do that from the folder `$WORK/../commun/Claire`,
so that all foundation models can be found in `$WORK/../commun/Claire/checkpoints`.

### Download raw data

Raw data can be found on 🤗 Hugging Face:
[OpenLLM-France/Claire-Dialogue-French-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1)

### Prepare data

Data preparation consists in tokenization, data augmentation, chunking/padding and conversion to binary format.

The script to generate the training data is `prepare_data.py`.
It takes as input a folder containing the raw data, and a folder containing the foundation model.
It generates a folder containing the training data, in the binary format to train efficiently.

By default, the script:
* generate multiple of 8 files for each dataset, to be able to use 1, 2, 4, or 8 GPUs.
  Use option `--multiple_of` to change that.
* groups datasets by type based on this [claire_data_groups.json](data/claire_data_groups.json). Use option `--group_datasets_by_genre` to change that.
* use padding to pack variable-length sequences. Use `--padding false` to put all sequences in a row.
* split too long sequences in chunks of `max_length` tokens (the maximum context window for the model), and try to split by starting at a speech turn.

Typical usage:
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
TRAINING_DIR=$SCRATCH/../commun/Claire/Claire-Falcon-7B-0.1

python pretrain.py \
--data_dir       $DATA_DIR \
--checkpoint_dir $FOUNDATION_MODEL_DIR \
--out_dir        $TRAINING_DIR \
--devices 8 \
--language fr \
--precision bf16-true \
--micro_batch_size 12 \
--batch_size 16 \
--num_epochs 1000 \
--max_checkpoints 20 \
--save_interval 1800 \
--eval_interval 1800 \
--enable_validation true \
--early_stopping 2 \
```

Note that the `--batch_size` option has to be tuned accordingly to the number of GPU devices.
The actual batch size will be `--batch_size` × `--devices`. We recommand an actual batch size around 130.
Option `--micro_batch_size` has to be tuned accordingly to the available GPU memory.

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

Training checkpoints and monitoring log can be found under output folder (`--out_dir`),
standard output and error should be recorded in that same folder.

### Monitoring

#### Offline validation

The script `validate_pretrain.py` can be used to validate a trained model on each dataset separately,
possibly while the model is training.

It will generate a csv file with the validation results (or append to an existing one).

```bash
TRAINING_DIR=$SCRATCH/../commun/Claire/pretrain/Claire-Falcon-7B-0.1

srun --ntasks=1 --gres=gpu:1 -C a100 --qos=qos_gpu-dev \
--output=$OUTDIR/validation.out --error=$OUTDIR/validation.out \
--time=00:10:00 \
python validate_pretrain.py \
--out_dir  $TRAINING_DIR \
--language fr \
--max 40 --batch_size 8
```

#### Convergence curves

The script `plot_convergence_curves.py` can be used to plot the evaluation of
the losses for online training (and offline validations if any).
You can give it one or several training folders (for comparison).

It will generate a plot like this:
![Convergence Curves of continual pretraining from Falcon-7b and Mistral-7B](training_history/ConvergenceCurve_0.1_Claire-Falcon-VS-Mistral.png)

## Check the model and make it available

### Merge LoRA weights

If trained with LoRA, you can first merge the weights, with a command like:
```bash
MODEL=tiiuae/falcon-7b
TRAINING_DIR=$SCRATCH/../commun/Claire/pretrain/Claire-Falcon-7B-0.1
TRAINED_MODEL_PATH=$TRAINING_DIR/iter-020883-ckpt.pth  # lit_model_lora_finetuned.pth
FOUNDATION_MODEL_DIR=$WORK/../commun/Claire/checkpoints/$MODEL
SAVE_DIR=$WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-0.1

srun --ntasks=1 --gres=gpu:1 --constraint=a100 \
python utils/merge_lora.py \
    --lora_path       $TRAINED_MODEL_PATH \
    --checkpoint_dir  $FOUNDATION_MODEL_DIR \
    --save_path       $SAVE_DIR/lit_model.pth \
```
This generates the merged model `lit_model.pth` in the specified `$SAVE_DIR`.

### Quick test of the model

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

### Convert trained Lit-GPT model and upload it to 🤗 Hugging Face

You can convert the model to [`transformers`](https://github.com/huggingface/transformers) with the following command.
```bash
TRAINING_DIR=$WORK/../commun/Claire/Claire-Falcon-7B-0.1
TRAINED_MODEL_PATH=$TRAINING_DIR/iter-020883-ckpt.pth
SAVE_DIR=$WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B-0.1

python convert_litgpt_to_transformers.py \
    --input_path $TRAINED_MODEL_PATH \
    --output_dir $SAVE_DIR \
    --repo_id    OpenLLM-France/Claire-7B-0.1
```
Use option `--repo_id` if and only if you want to upload the model to Hugging Face.
In this case, you will need to provide your [User Access Tokens](https://huggingface.co/settings/tokens).

The steps done by this script are:
* Copy relevant files from the foundation model checkpoint folder
  (This folder should be in `$TRAINING_DIR/hparams.json`, and can also be specified with option `--checkpoint_dir`)
* If needed, merge LoRA weights
* Convert the model in [lit-gpt](lit_gpt/scripts/convert_lit_checkpoint.py) format (`lit_model.pth`) to a model in the [transformers](https://github.com/huggingface/transformers) format (`pytorch_model.bin`).
* If needed, split the big model into chunks of <10 GB (ex: `pytorch_model-00001-of-00002.bin`, `pytorch_model-00002-of-00002.bin`, `pytorch_model.bin.index.json`)
* If asked (with `--repo_id`):
  * Create the Hugging Face repo if it does not exist
  * Upload the model and its companion files


#### Update Hugging Face model card

The model card ([README.md](hf_files/Claire-Falcon-7B-0.1/README.md)),
files requided to make an endpoint ([handler.py](hf_files/common/handler.py), [requirements.txt](hf_files/common/requirements.txt)),
and/or model files
can be updated on a Hugging Face model hub page (like [OpenLLM-France/Claire-7B-0.1](https://huggingface.co/OpenLLM-France/Claire-7B-0.1)) with the following command:
```bash
python utils/hf_upload_model.py \
OpenLLM-France/Claire-7B-0.1 \
--input_dir $DIR \
--message "<<your commit message>>"
```
where `$DIR` can be something like `hf_files/Claire-Falcon-7B-0.1`, `hf_files/common` or the folder that contains model weights (`pytorch_model*bin`).

You will need to provide your [HuggingFace User Access Tokens](https://huggingface.co/settings/tokens).


### Quantize the model (GGUF format)

Install [llama.cpp](https://github.com/ggerganov):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
python3 -m pip install -r requirements.txt
```

Run the conversion of the model packaged for Hugging Face,
choosing the desired quantization method:
```bash
# Convert to GGUF FP16 format
python3 convert.py /path/to/model/

# Quantize model weights
./quantize /path/to/model/ggml-model-f16.gguf /path/to/model/ggml-model-q4_0.gguf q4_0
```

Note: if you downloaded the model from Hugging Face,
it can be found by default under `.cache/huggingface/hub/models--OpenLLM-France--Claire-7B-0.1`.


## Acknowledgements

* [Lightning](https://github.com/Lightning-AI/lightning):  Deep Learning framework to train and deploy neural networks.
* [Lit-GPT](https://github.com/Lightning-AI/lit-gpt): Hackable implementation of state-of-the-art open-source Large Language Models.
* [Hugging Face](https://huggingface.co/models): Model hub containing state-of-the-art open-source Large Language Models.

This work was granted access to the HPC resources of IDRIS under the allocation 2023-AD011014561 made by GENCI
