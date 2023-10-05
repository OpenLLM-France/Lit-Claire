from huggingface_hub import login, HfApi, hf_hub_download
login()
from pathlib import Path


def upload_folder(folder_path: Path, repo_id: str):
    # download missing files ignored by lit_gpt/scripts/download.py

    revision="f7796529e36b2d49094450fb038cc7c4c86afa44"
    hf_hub_download(repo_id="tiiuae/falcon-7b", filename=".gitattributes", revision=revision)
    hf_hub_download(repo_id="tiiuae/falcon-7b", filename="README.md", revision=revision)
    hf_hub_download(repo_id="tiiuae/falcon-7b", filename="config.json", revision=revision)
    hf_hub_download(repo_id="tiiuae/falcon-7b", filename="configuration_RW.py", revision=revision)
    hf_hub_download(repo_id="tiiuae/falcon-7b", filename="modelling_RW.py", revision=revision)
    hf_hub_download(repo_id="tiiuae/falcon-7b", filename="special_tokens_map.json", revision=revision)

    api = HfApi()
    api.upload_folder(
        folder_path="/path/to/local/model",
        repo_id="username/my-cool-model",
        repo_type="model",
        ignore_patterns=["lit_*", "pytorch_model.bin.index.json"],
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_folder)
