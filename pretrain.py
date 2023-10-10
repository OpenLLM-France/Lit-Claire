import time
t_last_checkpoint = time.perf_counter()
t_last_valid = t_last_checkpoint

import os
import sys
import json
import math
import shutil
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader

# support running without installing as a package
this_folder = Path(__file__).parent.resolve()
sys.path = [str(this_folder / "lit_gpt")] + sys.path # Prepend to PYTHONPATH

from lit_gpt import Config, GPT
from lit_gpt.lora import Config as LoraConfig, GPT as LoraGPT
from lit_gpt.lora import Block, lora_filter, mark_only_lora_as_trainable
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from lightning.fabric.loggers import CSVLogger

from utils.data import create_dataloaders

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

    # Data
    try_small: bool = False,
    enable_validation: bool = True,

    # Action to be taken per n interval
    save_interval: int = 3540, # A little bit less than 1H
    eval_interval: int = 3540, # A little bit less than 1H
    log_interval: int = 1,
    interval_unit: str = "time", # "time" or "step"

    # Number of epochs
    num_epochs: int = 1,
    max_checkpoints: Optional[int] = None,
    early_stopping: Optional[int] = None, # When validation is enabled, number of validation steps without improvement before stopping

    # Batch
    batch_size: int = 192,
    micro_batch_size: int = 12,

    # Learning rate
    learning_rate: float = 1e-4,
    warmup_steps: int = 50,  # note: this is based on step, not iteration
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,

    # LORA
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = True,
    lora_value: bool = True,
    lora_projection: bool = True,
    lora_mlp: bool = True,
    lora_head: bool = True,
):
    hparams = dict((k,v) for k,v in locals().items())

    assert interval_unit in ["time", "steps"]

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

    logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=devices, accelerator=accelerator, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)

    fabric.launch(main, checkpoint_dir, out_dir, data_dir, try_small, enable_validation, hparams)

def main(fabric, checkpoint_dir, out_dir, data_dir, try_small, enable_validation, hparams):
    language            = hparams["language"]
    batch_size          = hparams["batch_size"]
    micro_batch_size    = hparams["micro_batch_size"]
    num_epochs          = hparams["num_epochs"]
    learning_rate       = hparams["learning_rate"]
    weight_decay        = hparams["weight_decay"]
    use_lora            = hparams["use_lora"]

    assert batch_size % micro_batch_size == 0 and batch_size > 0 and micro_batch_size > 0
    hparams["gradient_accumulation_iters"] = batch_size // micro_batch_size

    check_valid_checkpoint_dir(checkpoint_dir)  # check if there is lit-gpt format model

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    # Make output folder and copy source code and hyperparameters to out_dir
    os.makedirs(out_dir / "src", exist_ok=True)
    for file in __file__, "prepare_data.py", "data/claire_metadata.csv":
        shutil.copy2(this_folder / file, out_dir / "src" / os.path.basename(file))
    for folder in "lit_gpt/lit_gpt", "utils", :
        shutil.copytree(this_folder / folder, out_dir / "src" / folder,
            ignore=lambda x, y: ["__pycache__"], dirs_exist_ok=True)
    json.dump(
        # hparams : Path are converted to string because Path is not JSON serializable
        {k:str(v) if isinstance(v, Path) else v for k,v in hparams.items()},
        open(out_dir / "hparams.json", "w"),
        indent=2, ensure_ascii=False
    )
    
    if use_lora:
        lora_config = {k.split("lora_")[1]: v for k, v in hparams.items() if k.startswith("lora_")}
        lora_config = {(k if k in ["r", "alpha", "dropout"] else "to_"+k): v for k, v in lora_config.items()}
        config = LoraConfig.from_json(
            path=checkpoint_dir / "lit_config.json",
            **lora_config
            # r=lora_r, alpha=lora_alpha, dropout=lora_dropout, to_query=lora_query, to_key=lora_key, ...
        )
        with open(out_dir / "lora_config.json", "w") as file:
            json.dump(lora_config, file)
    else:
        config = Config.from_json(path=checkpoint_dir / "lit_config.json")

    (train_dataloader, train_details), (val_dataloader, val_details) = create_dataloaders(
        path=data_dir,
        language=language,
        batch_size=micro_batch_size,
        shuffle=True,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
        seed=(1337 + fabric.global_rank),
        verbose=True,
        try_small=try_small,
        max_validation_samples=200 if try_small else 4000,
        return_details=True,
        enable_validation=enable_validation,
    )

    max_train_iters = int(num_epochs * train_details["epoch_size"] // micro_batch_size)
    max_eval_iters = int(math.ceil(val_details["epoch_size"] // micro_batch_size))
    fabric.print(f"max train iters: {max_train_iters}")
    fabric.print(f"max eval iters: {max_eval_iters}")

    if try_small:
        fabric.print("QUICK TEST")

        # Reduce max values
        small_max_batches = 2 * batch_size // micro_batch_size
        fabric.print(f"* {max_train_iters=} -> {small_max_batches}")
        max_train_iters = small_max_batches
        fabric.print(f"* {max_eval_iters=} -> {small_max_batches}")
        max_eval_iters = small_max_batches

        # Reduce intervals
        for interval in ["eval_interval", "save_interval", "log_interval"]:
            fabric.print(f"* {interval=} -> 1")
            hparams[interval] = 1

    hparams["max_train_iters"] = max_train_iters
    hparams["max_eval_iters"] = max_eval_iters

    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=True):
        if use_lora:
            model = LoraGPT(config)
            mark_only_lora_as_trainable(model)
        else:
            model = GPT(config)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    sys.stdout.flush()

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)  # set foreach=False may reduce peak vram
    optimizer = fabric.setup_optimizers(optimizer)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=not use_lora)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(
        fabric, model, optimizer, train_dataloader, val_dataloader, speed_monitor, out_dir,
        hparams
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"
    save_checkpoint(fabric, model, save_path, use_lora=use_lora)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitor,
    out_dir: Path,
    hparams: dict,
    sanity_check: bool = False,
) -> None:
    micro_batch_size            = hparams["micro_batch_size"]
    gradient_accumulation_iters = hparams["gradient_accumulation_iters"]
    max_eval_iters              = hparams["max_eval_iters"]
    max_train_iters             = hparams["max_train_iters"]
    warmup_steps                = hparams["warmup_steps"]
    learning_rate               = hparams["learning_rate"]
    grad_clip                   = hparams["grad_clip"]
    save_interval               = hparams["save_interval"]
    eval_interval               = hparams["eval_interval"]
    log_interval                = hparams["log_interval"]
    interval_unit               = hparams["interval_unit"]
    max_checkpoints             = hparams["max_checkpoints"]
    early_stopping              = hparams["early_stopping"]
    use_lora                    = hparams["use_lora"]

    global t_last_checkpoint, t_last_valid
    num_checkpoints = 0
    best_valid_loss = float("inf")
    best_valid_loss_iter = 0
    valid_loss_iter = 0

    if val_dataloader is not None and sanity_check:
        sanity_check_val_loss = validate(fabric, model, val_dataloader, max_eval_iters=1)
        fabric.print(f"sanity check val loss: {sanity_check_val_loss.item():.4f}")

    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_lora_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, model.max_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num, train_data in enumerate(train_dataloader):
        if iter_num >= max_train_iters:
            break
        
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous()
        targets = train_data[:, 1 : model.max_seq_length + 1].contiguous()

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)  # set lm_head_chunk_size=128 may reduce peak vram
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)  # set chunk_size=128 may reduce peak vram
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )
            fabric.logger.log_metrics({"loss": f"{loss.item():.4f}"})

        if not is_accumulating:

            if iter_num == (max_train_iters-1):
                # Save and validate at the end of training
                condition_checkpoint = True
                condition_eval = True
            elif interval_unit == "time":
                # Save and validate at regular time intervals
                t = time.perf_counter()
                condition_checkpoint = (t - t_last_checkpoint) > save_interval
                condition_eval = condition_checkpoint or (t - t_last_valid) > eval_interval
            else:
                # Save and validate at regular iteration intervals
                condition_checkpoint = step_count % save_interval == 0
                condition_eval = step_count % eval_interval == 0

            if condition_checkpoint:
                t_last_checkpoint = time.perf_counter()
                checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
                save_checkpoint(fabric, model, checkpoint_path, use_lora=use_lora)
                num_checkpoints += 1
                if max_checkpoints and num_checkpoints >= max_checkpoints:
                    fabric.print(f"Reached max checkpoints: {max_checkpoints}")
                    break

            if condition_eval and val_dataloader is not None:
                t_last_valid = time.perf_counter()
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader, max_eval_iters)
                t1 = time.perf_counter() - t0
                speed_monitor.eval_end(t1)
                fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
                fabric.logger.log_metrics({"val_loss": f"{val_loss.item():.4f}"})
                fabric.barrier()
                if fabric.device.type == "cuda":
                    fabric.logger.log_metrics({"peak_vram": f"{torch.cuda.max_memory_allocated() / 1e9:.02f} GB"})

                valid_loss_iter += 1
                if val_loss <= best_valid_loss:
                    best_valid_loss = val_loss
                    best_valid_loss_iter = valid_loss_iter
                elif early_stopping and (valid_loss_iter - best_valid_loss_iter) >= early_stopping:
                    fabric.print(f"Early stopping at iter {iter_num}")
                    break


@torch.inference_mode()
def validate(
    fabric: L.Fabric, model: GPT, val_dataloader: DataLoader, max_eval_iters: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(max_eval_iters, device=fabric.device)
    for eval_iter_num, val_data in enumerate(val_dataloader):
        if eval_iter_num >= max_eval_iters:
            break
        input_ids = val_data[:, 0 : model.max_seq_length].contiguous()
        targets = val_data[:, 1 : model.max_seq_length + 1].contiguous()
        logits = model(input_ids)  # set lm_head_chunk_size=128 may reduce peak vram
        losses[eval_iter_num] = chunked_cross_entropy(logits, targets, chunk_size=0)  # set chunk_size=128 may reduce peak vram
    
    if eval_iter_num < max_eval_iters - 1:
        losses = losses[: eval_iter_num + 1]
    val_loss = losses.mean()

    model.train()
    return val_loss


def save_checkpoint(fabric, model, file_path: Path, use_lora: bool):
    if use_lora:
        fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
        fabric.save(file_path, {"model": model}, filter={"model": lora_filter})
    else:
        fabric.print(f"Saving weights to {str(file_path)!r}")
        fabric.save(file_path, {"model": model})

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
