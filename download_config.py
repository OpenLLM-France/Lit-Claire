from pathlib import Path
from huggingface_hub import hf_hub_download


def download(folder_path: Path="folder_path", repo_id: str="tiiuae/falcon-7b"):
    # download missing files which was not downloaded by lit_gpt/scripts/download.py
    revision="898df1396f35e447d5fe44e0a3ccaaaa69f30d36"
    filenames = ["README.md", "config.json", "special_tokens_map.json"]
    for filename in filenames:
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=folder_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download)
