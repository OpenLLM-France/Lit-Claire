import re
import csv
import sys
import random
import json
from pathlib import Path
import re
import os

import numpy as np
from tqdm import tqdm

from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path = [str(wd / "lit_gpt")] + sys.path # Prepend to PYTHONPATH

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer

from utils.metadata import get_metadata, metadata_filename_extra
from utils.text import augmented_texts_generator


###############
# Main function

def prepare_fn(
    source_path: Path, checkpoint_dir: Path, destination_path: Path,
    chunk_size: int,
    max_length: int = None,
    bos=None,
    eos=None,
    padding=True,
    filename_regex="full.txt",
    update_metadata=False,
) -> None:
    """Prepare the dataset using the tokenizer."""
    destination_path = destination_path.resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    if bos is None:
        bos = tokenizer.use_bos
        assert bos == tokenizer.check_if_bos_token_used(checkpoint_dir)
    if eos is None:
        eos = bool(tokenizer.eos_id)

    print(f"Using: {bos=}, {eos=}, {max_length=}")

    if not max_length:
        update_metadata = False
    if update_metadata:
        if os.path.isfile(metadata_filename_extra):
            metadata = list(csv.DictReader(open(metadata_filename_extra)))
        else:
            metadata = []
        metadata_dict = {row["dataset"]: row for row in metadata}
        key_convs_augmented = f"conversations_augmented"
        key_segments_augmented = f"segments_augmented_{max_length}"
        key_segments = f"segments_{max_length}"

    # First collect all files to process (making preliminary checks)
    all_files = {}
    for root, dirs, files in os.walk(source_path, followlinks=True):
        root = os.path.realpath(root)
        for file in files:
            if re.match(filename_regex + r"$", file):
                filepath = os.path.join(root, file)
                metadata = get_metadata(filepath)
                all_files[filepath] = metadata

    if len(all_files) == 0:
        raise RuntimeError(
            f"No input files found at {source_path}."
        )

    for filepath, metadata in all_files.items(): # tqdm(all_files.items(), unit="dataset"):
        set_name = metadata["dataset"]
        num_conversations = int(metadata["conversations"])
        prefix = set_name.replace("/", "--")
        print(f"Processing {filepath} -> {destination_path}/{prefix}*")

        dataset_hf = load_dataset("text", data_files={"train": filepath}, sample_by="paragraph", streaming=True)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        num_cuts = 0
        num_convs_augmented = 0
        num_segments_augmented = 0
        num_segments = 0
        min_len = 1e10
        max_len = 0
        for sample in tqdm(dataset_hf["train"], total=num_conversations, unit="conversations", desc=prefix):
            text = sample["text"]

            # Text normalization and augmentation
            for ivariant, text_variant in enumerate(augmented_texts_generator(text)):

                # # Uncomment for debugging of text augmentation
                # if ivariant > 0:
                #     if ivariant == 1:
                #         print(text.replace("\n", " ")[:100])
                #     print(text_variant.replace("\n", " ")[:100])

                text_ids = tokenizer.encode(text_variant, bos=bos, eos=eos)
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
                        if ivariant == 0:
                            num_segments += 1
                        num_segments_augmented += 1
                    num_cuts += 1
                else:
                    a = np.array(text_ids, dtype=builder.dtype)
                    if padding and len(a) < max_length:
                        a = np.pad(a, (0, max_length - len(a)), mode="constant", constant_values=tokenizer.eos_id)
                    min_len = min(min_len, len(a))
                    max_len = max(max_len, len(a))
                    builder.add_array(a)
                    if ivariant == 0:
                        num_segments += 1
                    num_segments_augmented += 1
                num_convs_augmented+= 1

        builder.write_reminder()

        print(f"* {num_cuts}/{num_convs_augmented} text cutted in several chunks")
        print(f"* min-max length: {min_len} - {max_len}")

        if update_metadata:
            metadata_dict[set_name] = metadata_dict.get(set_name, {}) | {
                "dataset": set_name,
                key_segments: num_segments,
                key_convs_augmented: num_convs_augmented,
                key_segments_augmented: num_segments_augmented,
            }
            
            # Update metadata file
            with open(metadata_filename_extra,"w") as file:
                metadata = list(metadata_dict.values())
                fieldnames = list(metadata[0].keys())
                for field in key_convs_augmented, key_segments, key_segments_augmented:
                    if field not in fieldnames:
                        fieldnames.append(field)
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata)

def prepare(
    source_path: Path = Path("data/source_data_folder"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    destination_path: Path = Path("data/prepared_data_folder"),
    padding: bool = True,
    update_metadata: bool = False,
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
        chunk_size=(config.block_size + 1) * 512,  # block size + 1 for causal, 512 blocks
        max_length=max_length,
        padding=padding,
        update_metadata=update_metadata,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
