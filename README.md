# CLAIRE

[Anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh)


# create environment
conda create -y -n claire python=3.8
conda activate claire
pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
pip install --user --no-cache-dir -r requirements.txt

# download Falcon-7b
python lit_gpt/scripts/download.py --repo_id tiiuae/falcon-7b
python lit_gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/tiiuae/falcon-7b


