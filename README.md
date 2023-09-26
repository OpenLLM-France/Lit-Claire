# CLAIRE

# download code
```
git clone --recurse-submodules https://github.com/OpenLLM-France/Claire
```

# create environment
[Anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh) version for local running.
```
module load cpuarch/amd
module load anaconda-py3/2023.03
```

```
conda create -y -n claire python=3.10
conda activate claire
pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
pip install --user --no-cache-dir -r requirements.txt
```

# download Falcon-7b
```
python lit_gpt/scripts/download.py --repo_id tiiuae/falcon-7b
python lit_gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

# prepare data
```
python prepare_data.py \
    --source_path       $WORK/../commun/Corpus_text/MULTILANG/OpenLLM \
    --checkpoint_dir    $WORK/../commun/Claire/checkpoints/tiiuae/falcon-7b \
    --destination_path  $SCRATCH/../commun/preprocessed_data/Claire/falcon-7b
```

