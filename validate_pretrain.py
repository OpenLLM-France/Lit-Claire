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
from utils.merge_lora import merge_lora
from utils.data import create_dataloaders

import torch
import lightning as L
from lit_gpt.model import Config, GPT, Block
from lit_gpt.lora import GPT as LoraGPT, Config as LoraConfig, Block as LoraBlock
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)
from lightning.fabric.strategies import FSDPStrategy
from lit_gpt.tokenizer import Tokenizer

def setup(
    # Folders
    out_dir: Path = Path("out/lora/Claire"),
    data_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    out_file: Optional[Path] = None,
    language: Optional[str] = None,

    # Hardware (only used in setup, not main)
    devices: int = 1,  # num_gpus_per_node
    num_nodes: int = 1,
    precision: Optional[str] = None,

    strategy: Optional[str] = "auto",
    batch_size: int = 12,

    try_small: bool = False,
    max_eval_iters: Optional[int] = None,

    debug: bool = False,
):
    hparams = dict((k,v) for k,v in locals().items())

    precision = precision or get_default_supported_precision(training=False)

    if devices > 1 or num_nodes > 1:
        raise NotImplementedError("Multi-node offline validation not supported yet")

    if out_file is None:
        out_file = out_dir / f"validation_results_{precision}_{strategy}.csv"

    use_lora = os.path.isfile(out_dir / "lora_config.json")

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={LoraBlock if use_lora else Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.print(hparams)

    fabric.launch(main, checkpoint_dir, out_dir, out_file, data_dir, try_small, hparams)

def main(fabric, checkpoint_dir, out_dir, out_file, data_dir, try_small, hparams):
    language            = hparams["language"]
    batch_size          = hparams["batch_size"]
    max_eval_iters0     = hparams["max_eval_iters"]
    use_lora            = os.path.isfile(out_dir / "lora_config.json")
    debug               = hparams["debug"]

    assert os.path.isdir(out_dir), f"Output directory {out_dir} does not exist."

    hparams = out_dir / "hparams.json"
    if not hparams.exists():
        hparams = None
    else:
        with open(hparams, "r") as f:
            hparams = json.load(f)

    if checkpoint_dir is None:
        if hparams is None: raise FileNotFoundError(f"Cannot find hyperparameter file {out_dir}/hparams.json")
        assert "checkpoint_dir" in hparams, f"Cannot find 'checkpoint_dir' in {hparams}"
        checkpoint_dir = Path(hparams["checkpoint_dir"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Cannot find {checkpoint_dir}")

    if data_dir is None:
        if hparams is None: raise FileNotFoundError(f"Cannot find hyperparameter file {out_dir}/hparams.json")
        assert "data_dir" in hparams, f"Cannot find 'data_dir' in {hparams}"
        data_dir = Path(hparams["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find {data_dir}")

    checkpoints = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".pth") and f.startswith("iter-")]
    assert len(checkpoints) > 0, f"No checkpoints found in {out_dir}"    
    checkpoints = sorted(checkpoints, key=lambda x: get_iter_info(x)["iter"], reverse=True)

    if os.path.isdir(out_dir / "src"):
        for file in __file__, os.path.join(this_folder, "utils", "merge_lora.py"), :
            shutil.copy2(file, out_dir / "src" / os.path.basename(file))

    check_valid_checkpoint_dir(checkpoint_dir)  # check if there is lit-gpt format model

    with fabric.init_module(empty_init=False):
        if use_lora:
            model = None
            # lora_config = json.load(open(out_dir / "lora_config.json", "r"))
            # config = LoraConfig.from_json(path=checkpoint_dir / "lit_config.json", **lora_config)
            # model = LoraGPT(config)
        else:
            config = Config.from_json(path=checkpoint_dir / "lit_config.json")
            model = GPT(config)

    _, (val_dataloaders, val_details) = create_dataloaders(
        path=data_dir,
        language=language,
        batch_size=batch_size,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
        seed=(1337 + fabric.global_rank),
        verbose=True,
        try_small=try_small,
        shuffle=True,
        max_validation_samples=200 if try_small else 4000,
        return_details=True,
        wrap_validation=False,
        split_validation_in_subsets=True,
        enable_train=False,
    )

    already_done = {}
    valid_file_exists = False
    if os.path.isfile(out_file):
        with open(out_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                valid_file_exists = True
                if str(row["max_iters"]) == str(max_eval_iters0):
                    name_model = row["file"]
                    dataset_name = row["data"]
                    already_done[name_model] = already_done.get(name_model, []) + [dataset_name]

    sys.stdout.flush()

    tokenizer = Tokenizer(checkpoint_dir) if debug else None

    with open(out_file, "a") as file:
        logger = None

        for checkpoint_path in checkpoints:
            info = get_iter_info(checkpoint_path)
            # if info["file"] in already_done:
            #     print(f"Skipping {info['file']} as it is already in the file")
            #     continue

            has_loaded_model = False

            for val_dataloader, val_detail in zip(val_dataloaders, val_details):

                dataset_name = val_detail["name"]
                if dataset_name in already_done.get(info["file"], []):
                    print(f"Skipping {info['file']} on {dataset_name} as it is already in the file")
                    continue

                if not has_loaded_model:
                    has_loaded_model = True
                    if use_lora:
                        model = merge_lora(
                            lora_path=Path(checkpoint_path),
                            checkpoint_dir=Path(checkpoint_dir),
                            model=None,
                            fabric=fabric,
                        )
                    else:
                        load_checkpoint(fabric, model, checkpoint_path, strict=not use_lora)

                    model = fabric.setup_module(model)
                    model.eval()

                val_dataloader = fabric.setup_dataloaders(val_dataloader)
                if max_eval_iters0 is None:
                    max_eval_iters = int(math.ceil(val_detail["epoch_size"] // batch_size))
                else:
                    max_eval_iters = max_eval_iters0

                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader, max_eval_iters=max_eval_iters, tokenizer=tokenizer)
                t1 = time.perf_counter() - t0
                info.update({
                    "data": dataset_name,
                    "loss": val_loss, # round(val_loss, 4 ) # f"{val_loss:.4f}",
                    "time": f"{t1:.3f} sec",
                    "batch_size": batch_size,
                    "max_iters": max_eval_iters0,
                })
                if fabric.device.type == "cuda":
                    info.update({"peak_vram": f"{torch.cuda.max_memory_allocated() / 1e9:.02f} GB"})

                fabric.print(json.dumps(info, indent=4))
                if logger is None:
                    logger = csv.DictWriter(file, fieldnames=info.keys(), lineterminator='\n')
                    if not valid_file_exists:
                        logger.writeheader()
                logger.writerows([info])

                fabric.barrier()
                sys.stdout.flush()
                file.flush()

                # break

            # Test one checkpoint at a time to avoid bugs...
            # break

def get_iter_info(checkpoint_path):
    iter_num = int(os.path.basename(checkpoint_path).split("-")[1])
    return {"iter": iter_num, "file": os.path.basename(checkpoint_path)}


if __name__ == "__main__":
    from jsonargparse import CLI

    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    CLI(setup)
