from huggingface_hub import login, HfApi, hf_hub_download
login()
from pathlib import Path


def upload(folder_path: Path="folder_path", repo_id: str="repo_id", create_repo: bool=False):
    # download missing files which was not downloaded by lit_gpt/scripts/download.py
    revision="f7796529e36b2d49094450fb038cc7c4c86afa44"
    filenames = [".gitattributes", "README.md", "config.json", "configuration_RW.py", "modelling_RW.py", "special_tokens_map.json"]
    for filename in filenames:
        hf_hub_download(repo_id="tiiuae/falcon-7b", filename=filename, revision=revision, local_dir=folder_path)

    api = HfApi()

    if create_repo is True:
        api.create_repo(
            repo_id=repo_id,
            private=True,
            repo_type="model",
            exist_ok=False,
            )

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["lit_*", "pytorch_model.bin.index.json"],
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload)
