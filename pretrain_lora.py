import os
import sys
import time
import json
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path = [str(wd / "lit_gpt")] + sys.path # Prepend to PYTHONPATH

from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
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
# from utils.redpajama_data import create_dataloaders as create_dataloaders_redpajama


# Action to be taken per n iteration
eval_interval = 40
save_interval = 40
log_interval = 1

# Learning rate
learning_rate = 3e-4
warmup_steps = 2  # note: this is based on step, not iteration
weight_decay = 0.01

# Batch
batch_size = 8
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0

# Number of epochs
num_epochs = 3

# LORA
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    devices: int = 2,  # num_gpus_per_node
    num_nodes: int = 1,
    data_dir: Path = Path("data/preprocessed_data"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    out_dir: Path = Path("out/lora/Claire"),
    precision: Optional[str] = None,
    try_small: bool = False,
):
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
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, try_small)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, try_small: bool):
    check_valid_checkpoint_dir(checkpoint_dir)  # check if there is lit-gpt format model

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    os.makedirs(out_dir, exist_ok=True)
    
    if not any((lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    config = Config.from_json(
        path=checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )
    lora_config = {k.split("lora_")[1]: v for k, v in hparams.items() if k.startswith("lora_")}
    lora_config = {(k if k in ["r", "alpha", "dropout"] else "to_"+k): v for k, v in lora_config.items()}
    with open(out_dir / "lora_config.json", "w") as file:
        json.dump(lora_config, file)

    # DEPRECATED for debug
    # train_dataloader, val_dataloader = create_dataloaders_redpajama(
    #     batch_size=micro_batch_size,
    #     block_size=config.block_size,
    #     fabric=fabric,
    #     train_data_dir=data_dir,
    #     val_data_dir=data_dir,
    #     seed=(1337 + fabric.global_rank),
    # )

    (train_dataloader, train_details), (val_dataloader, val_details) = create_dataloaders(
        batch_size=micro_batch_size,
        path=data_dir,
        shuffle=True,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
        seed=(1337 + fabric.global_rank),
        verbose=True,
        try_small=try_small,
        return_details=True,
    )
    max_train_iters = num_epochs * train_details["epoch_size"] // micro_batch_size
    max_eval_iters = max(1, val_details["epoch_size"] // micro_batch_size)

    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)  # TODO: check foreach=False
    optimizer = fabric.setup_optimizers(optimizer)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(
        fabric, model, optimizer, train_dataloader, val_dataloader, out_dir, speed_monitor,
        max_train_iters=max_train_iters,
        max_eval_iters=max_eval_iters,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"  # TODO
    save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    max_eval_iters: int,
    max_train_iters: int,
    sanity_check: bool = True,
) -> None:
    if val_dataloader is not None and sanity_check:
        sanity_check_val_loss = validate(fabric, model, val_dataloader, max_eval_iters=1)
        fabric.print({"sanity check val loss:", f"{sanity_check_val_loss.item():.4f}"})

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
            logits = model(input_ids, lm_head_chunk_size=128)  # TODO: check lm_head_chunk_size
            loss = chunked_cross_entropy(logits, targets)  # TODO: check chunk_size
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            # TODO: fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
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

        if val_dataloader is not None and not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_eval_iters)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.logger.log_metrics({"val_loss": f"{val_loss.item():.4f}"})
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)


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
        logits = model(input_ids, lm_head_chunk_size=128)
        losses[eval_iter_num] = chunked_cross_entropy(logits, targets)
    val_loss = losses.mean()

    model.train()
    return val_loss


def save_lora_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
