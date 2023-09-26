import re
import csv
import sys
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd / "lit_gpt"))

import lit_gpt.lit_gpt.packed_dataset as packed_dataset
from lit_gpt.lit_gpt.config import Config
from lit_gpt.lit_gpt.tokenizer import Tokenizer


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
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int
) -> None:
    """Prepare the dataset using the tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    for row in WEIGHTS_CSV:
        set_name = row["lang"] + "-" + row["set_name"]
        file_name = "full.txt"

        filepath = source_path / row["lang"] / row["set_name"] / file_name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}."
            )

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {set_name}")

        dataset = load_dataset("text", data_files={"train": str(filepath)}, sample_by="paragraph", streaming=True)
        updated_dataset = dataset.map(augment_fn)
        for sample in tqdm(updated_dataset["train"]):
            text = sample["text"]
            text_ids = tokenizer.encode(text)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare(
    source_path: Path = Path("data/source_data_folder"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    destination_path: Path = Path("data/prepared_data_folder"),
    sample: bool = True,
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""
    config = Config.from_json(checkpoint_dir / "lit_config.json")

    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(config.block_size + 1) * 1024,  # block size + 1 for causal, 1024 blocks
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
