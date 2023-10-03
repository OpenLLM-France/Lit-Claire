# CLAIRE

### clone the repo
```
git clone --recurse-submodules https://github.com/OpenLLM-France/Claire
```

### create environment

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

Then, install the dependencies (on Jean Zay, you may want to use `--user` if you ):
```bash
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
pip install --no-cache-dir -r requirements.txt
```

### download then convert Hugging Face model to Lit-GPT format

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
- `srun python pretrain_lora.py --devices 2`
  
These 2 arguments should be equal:
- `#SBATCH --nodes=1`
- `srun python pretrain_lora.py --num_nodes 1`

training checkpoints and monitoring log can be found under `out_dir`, standard output and error are recorded in `pretrain_lora.out`

on Jean Zay, you can check the status of the job with `squeue -u $USER`
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
100813    gpu_p5 pretrain  ugs29jo  R       0:52      1 jean-zay-iam36
```
cancel the job with `scancel 100681`, connect to the node with `ssh jean-zay-iam36` (on which you can run `nvidia-smi`)
