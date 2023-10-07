from transformers import AutoModelForCausalLM
from pathlib import Path
from huggingface_hub import hf_hub_download


def split(folder_path: Path="folder_path"):
    # download missing files which was not downloaded by lit_gpt/scripts/download.py
    print("downloading config.json ...")
    revision="898df1396f35e447d5fe44e0a3ccaaaa69f30d36"
    filenames = ["README.md", "config.json", "special_tokens_map.json"]
    for filename in filenames:
        hf_hub_download(repo_id="tiiuae/falcon-7b", filename=filename, revision=revision, local_dir=folder_path)

    print("loading model ...")
    model = AutoModelForCausalLM.from_pretrained(folder_path)
    
    print("spliting model ...")
    model.save_pretrained(folder_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(split)
