import torch
from transformers import AutoModelForCausalLM
from pathlib import Path


def split(folder_path: Path="folder_path"):
    print("loading model ...")
    model = AutoModelForCausalLM.from_pretrained(folder_path, torch_dtype=torch.bfloat16)
    
    print("spliting model ...")
    model.save_pretrained(folder_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(split)
