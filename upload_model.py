from huggingface_hub import login, HfApi
login()
from pathlib import Path


def upload(folder_path: Path="folder_path", repo_id: str="repo_id", create_repo: bool=False):
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
        ignore_patterns=["lit_*", "pytorch_model.bin"],
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload)
