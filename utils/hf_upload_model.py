import os
import huggingface_hub
from pathlib import Path
from typing import Optional

wd = Path(__file__).parent.parent.resolve()


def upload_to_huggingface_hub(
    repo_id: str,
    input_dir: Path = wd / "hf_files",
    message: Optional[str] = None,
    create_repo: Optional[bool] = None,
):
    print(f"Uploading repository https://huggingface.co/{repo_id} with:\n" + "\n".join(os.listdir(input_dir)))
    huggingface_hub.login()

    api = huggingface_hub.HfApi()

    if create_repo is None:
        create_repo = False
        try:
            api.repo_info(repo_id)
        except huggingface_hub.utils.RepositoryNotFoundError:
            create_repo = True

    if create_repo:
        if not message:
            message = "initial commit"
        print(f"Creating repository https://huggingface.co/{repo_id}")
        api.create_repo(
            repo_id=repo_id,
            private=True,
            repo_type="model",
            exist_ok=False,
        )

    api.upload_folder(
        folder_path=input_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["lit_*", "pytorch_model.bin"],
        commit_message=message,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_to_huggingface_hub)
