import huggingface_hub
from pathlib import Path

def upload_to_huggingface_hub(input_dir: Path="input_dir", repo_id: str="repo_id", create_repo: bool=False):

    huggingface_hub.login()

    api = huggingface_hub.HfApi()

    try:
        api.repo_info(repo_id)
    except huggingface_hub.utils.RepositoryNotFoundError:
        print(f"Creating repository https://huggingface.co/{repo_id}")
        api.create_repo(
            repo_id=repo_id,
            private=True,
            repo_type="model",
            exist_ok=False,
        )

    print(f"Uploading repository https://huggingface.co/{repo_id}")
    api.upload_folder(
        folder_path=input_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["lit_*", "pytorch_model.bin"],
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_to_huggingface_hub)
