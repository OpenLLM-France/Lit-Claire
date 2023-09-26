import re
import csv
import sys
import random
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path = [str(wd / "lit_gpt")] + sys.path # Prepend to PYTHONPATH

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer


WEIGHTS_CSV = csv.DictReader(open(wd / "data" / "claire_weights.csv"))

def augment_fn(sample):
    # speaker anonymization and randomization
    speakers = set(re.findall(r"\[.+:\]", sample["text"]))
    anonymized_randomized_speakers = ["[speaker" + f"{i+1:03}" + ":]" for i in range(len(speakers))]
    random.shuffle(anonymized_randomized_speakers)

    for s, a_r_s in zip(speakers, anonymized_randomized_speakers):
        sample["text"] = sample["text"].replace(s, a_r_s)
    
    return sample


def prepare_fn(
    source_path: Path, checkpoint_dir: Path, destination_path: Path,
    chunk_size: int,
    max_length: int = None,
    bos=None,
    eos=None,
    padding=True,
) -> None:
    """Prepare the dataset using the tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    if bos is None:
        bos = tokenizer.use_bos
        assert bos == tokenizer.check_if_bos_token_used(checkpoint_dir)
    if eos is None:
        eos = bool(tokenizer.eos_id)

    print(f"Using: {bos=}, {eos=}, {max_length=}")

    for row in WEIGHTS_CSV:
        set_name = row["lang"] + "-" + row["set_name"]
        file_name = "full.txt"

        filepath = source_path / row["lang"] / row["set_name"] / file_name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}."
            )

        print(f"Processing {set_name}")

        dataset_hf = load_dataset("text", data_files={"train": str(filepath)}, sample_by="paragraph", streaming=True)
        dataset_hf = dataset_hf.map(augment_fn)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        num_cuts = 0
        num_total = 0
        min_len = 1e10
        max_len = 0
        for sample in tqdm(dataset_hf["train"]):
            text = sample["text"]
            text_ids = tokenizer.encode(text, bos=bos, eos=eos)
            if max_length and len(text_ids) > max_length:
                # Cut in several chunks
                for i in range(0, len(text_ids), max_length):
                    a =np.array(text_ids[i:i+max_length], dtype=builder.dtype)
                    if len(a) <= 10:
                        # Leave too short tails
                        continue
                    if padding and len(a) < max_length:
                        a = np.pad(a, (0, max_length - len(a)), mode="constant", constant_values=tokenizer.eos_id)
                    min_len = min(min_len, len(a))
                    max_len = max(max_len, len(a))
                    builder.add_array(a)
                num_cuts += 1
            else:
                a = np.array(text_ids, dtype=builder.dtype)
                if padding and len(a) < max_length:
                    a = np.pad(a, (0, max_length - len(a)), mode="constant", constant_values=tokenizer.eos_id)
                min_len = min(min_len, len(a))
                max_len = max(max_len, len(a))
                builder.add_array(a)
            num_total+= 1

        builder.write_reminder()

        print(f"* {num_cuts}/{num_total} text cutted in several chunks")
        print(f"* min-max length: {min_len} - {max_len}")


def prepare(
    source_path: Path = Path("data/source_data_folder"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    destination_path: Path = Path("data/prepared_data_folder"),
    padding: bool = True,
) -> None:
    """Prepare the "Claire" dataset. We assume tokenizer has been trained."""
    config = Config.from_json(checkpoint_dir / "lit_config.json")

    max_length = None
    tokenizer_config_file = checkpoint_dir / "tokenizer_config.json"
    if tokenizer_config_file.is_file():
        tokenizer_config = json.load(open(tokenizer_config_file))
        max_length = tokenizer_config["model_max_length"]

    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(config.block_size + 1) * 1024,  # block size + 1 for causal, 1024 blocks
        max_length=max_length,
        padding=padding,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
