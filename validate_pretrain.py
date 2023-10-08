import os
import sys
import json
import math
import time
import shutil
import csv
from pathlib import Path
from typing import Optional

this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_folder)

from pretrain import validate
from merge_lora import merge_lora
from utils.data import create_dataloaders

import torch
import lightning as L
from lit_gpt.lora import GPT as LoraGPT, Config as LoraConfig
from lit_gpt import GPT, Config
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)

def setup(
    # Folders
    data_dir: Path = Path("data/preprocessed_data"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    out_dir: Path = Path("out/lora/Claire"),
    language: Optional[str] = None,

    # Hardware (only used in setup, not main)
    devices: int = 2,  # num_gpus_per_node
    num_nodes: int = 1,
    precision: Optional[str] = None,

    batch_size: int = 12,

    try_small: bool = False,
    max_eval_iters: Optional[int] = None,
):
    hparams = dict((k,v) for k,v in locals().items())

    precision = precision or get_default_supported_precision(training=True)

    accelerator = "auto"
    if devices > 1 or num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
        if devices == 0: # CPU
            devices = 1 # Using more causes "libgomp: Thread creation failed: Resource temporarily unavailable"
            accelerator = "cpu"

    fabric = L.Fabric(devices=devices, accelerator=accelerator, num_nodes=num_nodes, strategy=strategy, precision=precision)
    fabric.print(hparams)

    fabric.launch(main, checkpoint_dir, out_dir, data_dir, try_small, hparams)

def main(fabric, checkpoint_dir, out_dir, data_dir, try_small, hparams):
    language            = hparams["language"]
    batch_size          = hparams["batch_size"]
    max_eval_iters      = hparams["max_eval_iters"]
    use_lora            = os.path.isfile(out_dir / "lora_config.json")

    assert os.path.isdir(out_dir), f"Output directory {out_dir} does not exist."

    checkpoints = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".pth") and f.startswith("iter-")]
    assert len(checkpoints) > 0, f"No checkpoints found in {out_dir}"    
    checkpoints = sorted(checkpoints, key=lambda x: get_iter_info(x)["iter"])

    if os.path.isdir(out_dir / "src"):
        shutil.copy2(__file__, out_dir / "src" / os.path.basename(__file__))

    check_valid_checkpoint_dir(checkpoint_dir)  # check if there is lit-gpt format model

    with fabric.init_module(empty_init=True):
        if use_lora:
            lora_config = json.load(open(out_dir / "lora_config.json", "r"))
            config = LoraConfig.from_json(path=checkpoint_dir / "lit_config.json", **lora_config)
            model = LoraGPT(config)
        else:
            config = Config.from_json(path=checkpoint_dir / "lit_config.json")
            model = GPT(config)

    _, (val_dataloader, val_details) = create_dataloaders(
        path=data_dir,
        language=language,
        batch_size=batch_size,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
        seed=(1337 + fabric.global_rank),
        verbose=True,
        try_small=try_small,
        max_validation_samples=200 if try_small else 4000,
        return_details=True,
        wrap_validation=False,
        enable_train=False,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    if max_eval_iters is None:
        max_eval_iters = int(math.ceil(val_details["epoch_size"] // batch_size))

    filename = out_dir / "validation_results.csv"

    already_done = []
    if os.path.isfile(out_dir / filename):
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                already_done.append(row["file"])

    with open(filename, "a") as file:
        logger = None

        for checkpoint_path in checkpoints:
            info = get_iter_info(checkpoint_path)
            if info["file"] in already_done:
                continue

            if use_lora:
                model = merge_lora(
                    checkpoint_dir=checkpoint_dir,
                    lora_dir=out_dir,
                    lora_pth_name=os.path.basename(checkpoint_path),
                    model=model,
                )
            else:
                load_checkpoint(fabric, model, checkpoint_path, strict=not use_lora)

            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_eval_iters=max_eval_iters)
            t1 = time.perf_counter() - t0
            info.update({"val_loss": f"{val_loss.item():.4f}", "val time": f"{t1 * 1000:.2f}ms"})
            if fabric.device.type == "cuda":
                info.update({"peak_vram": f"{torch.cuda.max_memory_allocated() / 1e9:.02f} GB"})

            fabric.print(json.dumps(info, indent=4))
            if logger is None:
                logger = csv.DictWriter(file, fieldnames=info.keys(), lineterminator='\n')
                if not already_done:
                    logger.writeheader()
            logger.writerows([info])
            fabric.barrier()

def get_iter_info(checkpoint_path):
    iter_num = int(os.path.basename(checkpoint_path).split("-")[1])
    return {"iter": iter_num, "file": os.path.basename(checkpoint_path)}


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
