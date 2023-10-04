"""This script merges the LoRA weights with the base model"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path = [str(wd / "lit_gpt")] + sys.path # Prepend to PYTHONPATH

from lit_gpt.lora import GPT, Config, lora_filter, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load


def merge_lora(
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    lora_dir: Path = Path("out/lora/Claire"),
    lora_pth_name: str = "lit_model_lora_finetuned.pth",
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        lora_dir: Path to the checkpoint folder with trained adapter weights, which are the output of `finetune/lora.py`
        lora_pth_name: File name of the lora weights
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)
    fabric = L.Fabric(devices=1, precision=precision)

    check_valid_checkpoint_dir(checkpoint_dir)

    with open(lora_dir / "lora_config.json", "r") as file:
        lora_config = json.load(file)
    config = Config.from_json(
        path=checkpoint_dir / "lit_config.json",
        **lora_config
    )

    lora_path = lora_dir / lora_pth_name
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(lora_path) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint)

    merge_lora_weights(model)

    save_path = checkpoint_dir.parent.parent / "OpenLLM-France" / "Claire-7b" / "lit_model.pth"
    os.makedirs(save_path.parent, exist_ok=False)
    fabric.print(f"Saving weights to {str(save_path)!r}")
    # remove lora parameters and the lora linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(merge_lora)
