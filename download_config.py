from pathlib import Path
from huggingface_hub import snapshot_download
from typing import Optional
import os


def download(checkpoint_dir: Path="checkpoint_dir", repo_id: str="tiiuae/falcon-7b", access_token: Optional[str] = os.getenv("HF_TOKEN")):
    # download missing files which was not downloaded by lit_gpt/scripts/download.py
    revision="898df1396f35e447d5fe44e0a3ccaaaa69f30d36"
    download_files = ["README.md", "config.json", "special_tokens_map.json"]
    snapshot_download(
        repo_id,
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=download_files,
        token=access_token,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download)
