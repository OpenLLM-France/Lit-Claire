import re
import csv
import sys
import random
import json
from pathlib import Path
import re
import os
import shutil
import filelock
import glob
import math
import torch

import numpy as np
from tqdm import tqdm

from datasets import load_dataset

# support running without installing as a package
this_folder = Path(__file__).parent.resolve()
sys.path.append(str(this_folder / "lit-gpt"))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer

from utils.metadata import get_metadata, metadata_filename_extra, accumulate_metadata_by_group
from utils.text import augmented_texts_generator


###############
# Main function

def prepare_fn(
    source_path: Path, checkpoint_dir: Path, destination_path: Path,
    effective_block_size: int = None,
    multiple_of: int = 8,
    group_datasets_by_genre: bool = True,
    bos=None,
    eos=None,
    padding=True,
    pad_id=-1,
    filename_full="full.txt",
    filename_train="train.txt",
    filename_dev="test.txt",
    skip_if_exists=True,
    update_metadata=False,
    cut_around_turns=True,
    update_weights=False,
    DEBUG_PRINT=False,
) -> None:
    """Prepare the dataset using the tokenizer."""
    destination_path = destination_path.resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    if bos is None:
        bos = tokenizer.use_bos
        assert bos == tokenizer.check_if_bos_token_used(checkpoint_dir)
    if eos is None:
        eos = True # bool(tokenizer.eos_id)

    print(f"Using: {bos=}, {eos=}, {effective_block_size=}")

    if not effective_block_size:
        update_metadata = False

    # First collect all files to process (making preliminary checks)
    all_files = {}
    for root, dirs, files in os.walk(source_path, followlinks=True):
        root = os.path.realpath(root)
        files_for_this_dataset = []
        if filename_train in files:
            files_for_this_dataset = [filename_train]
        elif filename_full in files:
            files_for_this_dataset = [filename_full]
        if filename_dev in files:
            assert filename_full not in files_for_this_dataset
            files_for_this_dataset += [filename_dev]
        for f in files_for_this_dataset:
            filepath = os.path.join(root, f)
            metadata = get_metadata(filepath)
            metadata["is_dev"] = (f == filename_dev)
            all_files[filepath] = metadata

    if group_datasets_by_genre:
        # Group files together
        list_of_files, metadatas = accumulate_metadata_by_group(all_files.keys(), all_files.values())
    else:
        list_of_files, metadatas = zip(*all_files.items())


    # Note: we need to avoid unsigned integers to be able to pass -1
    #      (and lit-gpt with dtype="auto" can choose np.uint16) 
    vocab_size = tokenizer.vocab_size
    if vocab_size is not None and vocab_size < 32768:
        dtype = np.int16
    else:
        dtype = np.int32

    # Get tokens around tags for turns
    example_speakers = ["[speaker001:]", "[Intervenant 1:]"] + ["[" + chr(o) + ":]" for o in list(range(65,90+1)) + list(range(48,57+1)) + list(range(192, 212+1))]  # A-Z0-9À...
    tag_tokens = [tokenizer.encode("a\n" + s, bos=False, eos=False) for s in example_speakers]
    dtype_torch = tag_tokens[0].dtype
    if cut_around_turns:
        tag_tokens = [t.tolist() for t in tag_tokens]
        tag_tokens_prefix = common_prefix(tag_tokens)
        tag_tokens_suffix = common_suffix(tag_tokens)
        assert len(tag_tokens_prefix) > 0, f"Weird tokenizer. Cannot find common prefix for {tag_tokens}"
        assert len(tag_tokens_suffix) > 0, f"Weird tokenizer. Cannot find common suffix for {tag_tokens}"
        for tokens, expected in [(tag_tokens_prefix, "a\n["), (tag_tokens_suffix, ":]")]:
            actual = tokenizer.decode(torch.tensor(tokens, dtype=dtype_torch))
            assert actual == expected, f"Unexpected tokenizer behaviour. got {actual} instead of {expected} ({tokens=})"
        if len(tag_tokens_prefix) != 3 or len(tag_tokens_suffix) > 2:
            raise NotImplementedError(f"Tokenizer not supported: {tag_tokens_prefix=}, {tag_tokens_suffix=}")
        tag_token_prefix = tag_tokens_prefix[-1]
        tag_token_suffix = tag_tokens_suffix[-1]

        no_prefix = torch.tensor([], dtype=dtype_torch)
        bos_seq = torch.tensor([tokenizer.bos_id], dtype=dtype_torch)

    if len(all_files) == 0:
        raise RuntimeError(f"No input files found at {source_path}.")

    for filepaths, metadata in zip(list_of_files, metadatas):
        if isinstance(filepaths, str): filepaths = [filepaths]
        assert len(filepaths) > 0

        set_name = metadata["dataset"]
        num_conversations = int(metadata["conversations"])
        is_spontaneous = metadata["spontaneous"]
        assert is_spontaneous in [True, False]
        augmentation_level = 8 if is_spontaneous else 1
        force_augmentation = True

        # Do not augment validation
        if metadata["is_dev"]:
            augmentation_level = 0
            force_augmentation = False

        prefix = set_name.replace("/", "--")
        metadata_filename = destination_path / f"{prefix}_metadata.json"

        if not metadata["is_dev"]:
            print(f"{metadata_filename}\n: sampling_rate = {metadata['sampling_rate']}")
        if update_weights:
            if os.path.isfile(metadata_filename):
                with open(metadata_filename) as f:
                    old_metadata = json.load(f)
                    old_metadata["sampling_rate"] = metadata["sampling_rate"]
                metadata_filename.write_text(json.dumps(old_metadata, indent=4))
            else:
                print(f"WARNING: {metadata_filename} not found. Sampling rate not updated.")

        filenames = glob.glob(f"{destination_path}/{prefix}*bin")
        if len(filenames) > 0:
            # Skip, or remove existing files, if any
            if skip_if_exists:
                print(f"Skipping {filepaths[0]} because {prefix}*bin files already exist")
                continue
            else:
                for fn in glob.glob(f"{destination_path}/{prefix}*"):
                    os.remove(fn)
        elif skip_if_exists:
            # Create a dummy file to avoid other processes to process the same dataset
            Path(f"{destination_path}/{prefix}_0000000000.bin").touch()

        print(f"Processing:\n{filepaths} -> {destination_path}/{prefix}*\n{augmentation_level=}")

        dataset_hf = load_dataset("text", data_files={"train": filepaths}, sample_by="paragraph", streaming=True)

        # First get the number of samples, then build files
        for build_it in False, True:

            if build_it:

                # Get the right number of chunks
                num_segments_per_file = math.ceil(num_segments_augmented / multiple_of)
                it = 1
                while num_segments_per_file > 512:
                    it += 1
                    num_segments_per_file = math.ceil(num_segments_augmented / (multiple_of * it))
                chunk_size = effective_block_size * num_segments_per_file
                
                num_files = math.ceil(num_segments_augmented/num_segments_per_file)
                num_padded = num_segments_per_file * num_files - num_segments_augmented
                print(f"Will cut in {num_files} files of {num_segments_per_file} samples each ({num_segments_augmented} + {num_padded} padded)")

                # Write metadata
                metadata.update({
                    "num_samples" : num_segments_augmented,
                    "num_samples_rounded" : num_segments_per_file * num_files,
                    "num_samples_per_file" : num_segments_per_file,
                    "num_files" : num_files,
                    "num_padded" : num_padded,
                    "block_size" : effective_block_size,
                })
                metadata_filename.write_text(json.dumps(metadata, indent=4))
            
            else:

                # Dummy value
                chunk_size = effective_block_size * 512

            # Set the builder
            builder = packed_dataset.PackedDatasetBuilder(
                outdir=destination_path,
                prefix=prefix,
                chunk_size=chunk_size,
                sep_token=pad_id,
                dtype=dtype,
                vocab_size=tokenizer.vocab_size,
            )

            # Init counters
            num_cuts = 0
            num_convs = 0
            num_convs_augmented = 0
            num_segments_augmented = 0
            num_segments = 0
            min_len = 1e10
            max_len = 0

            random.seed(num_conversations) # For deterministic text augmentation

            for sample in tqdm(dataset_hf["train"], total=num_conversations, unit="conversations", desc=f"{prefix} ({2 if build_it else 1}/2)"):
                text = sample["text"]

                try:
                    # Text normalization and augmentation
                    for ivariant, text_variant in enumerate(augmented_texts_generator(text, augmentation_level, force_augmentation=force_augmentation)):

                        # # Uncomment for debugging of text augmentation
                        # if ivariant > 0:
                        #     if ivariant == 1:
                        #         print(text.replace("\n", " ")[:100])
                        #     print(text_variant.replace("\n", " ")[:100])

                        text_ids = tokenizer.encode(text_variant, bos=bos, eos=eos)
                        if effective_block_size and len(text_ids) > effective_block_size:
                            add_prefix = no_prefix
                            # Cut in several chunks
                            istart = 0
                            while istart < len(text_ids) - 1: # "-1" to avoid having end token alone
                                iend = istart + effective_block_size - len(add_prefix)
                                assert iend > istart
                                selec = text_ids[istart:iend]
                                if len(add_prefix):
                                    selec = torch.cat([add_prefix, selec])
                                    if DEBUG_PRINT: print("=== with prefix ===\n", tokenizer.decode(selec))
                                elif DEBUG_PRINT: print("=== standard ===\n", tokenizer.decode(selec))
                                a =np.array(selec, dtype=dtype)
                                if not cut_around_turns and len(a) <= 10:
                                    # Leave too short tails
                                    break
                                if padding and len(a) < effective_block_size:
                                    a = np.pad(a, (0, effective_block_size - len(a)), mode="constant", constant_values=pad_id)
                                min_len = min(min_len, len(a))
                                max_len = max(max_len, len(a))
                                assert len(a) == effective_block_size
                                if build_it:
                                    builder.add_array(a)
                                if ivariant == 0:
                                    num_segments += 1
                                num_segments_augmented += 1

                                previous_istart = istart
                                
                                if not cut_around_turns:
                                    # Naive
                                    istart += effective_block_size
                                    continue

                                # Start with the last turn
                                selec = text_ids[istart:iend+20]
                                candidates_end = torch.where(selec == tag_token_suffix)[0].tolist()
                                if len(candidates_end) > 0:
                                    # Cut around the last end of turn tag
                                    candidates_start = torch.where(selec[:candidates_end[-1]] == tag_token_prefix)[0].tolist()
                                    if 0 not in candidates_start:
                                        candidates_start = [0] + candidates_start
                                    assert len(candidates_start)

                                    # There may be more "[" than ":]" because of special tokens ("[bruit]", ...)
                                    # hence the trick with the shift
                                    i = 1
                                    shift = 0
                                    new_start = None
                                    while i <= len(candidates_end):
                                        candidate_end = candidates_end[-i]
                                        while i+shift > 1 and candidates_start[-i-shift+1] < candidate_end:
                                            shift -= 1
                                        candidate_start = candidates_start[-i-shift]
                                        # print(f"DBG PRINT {candidate_start=},  {candidate_end=} {tokenizer.decode(selec[candidate_start:candidate_end+1])}")
                                        while candidate_start > candidate_end:
                                            shift += 1
                                            candidate_start = candidates_start[-i-shift]
                                        i += 1
                                        # If we are close to the end, try to start as soon as possible
                                        if new_start is None or candidate_start + istart > len(text_ids) - effective_block_size:
                                            new_start = candidate_start
                                        else:
                                            break
                                    assert new_start is not None

                                    if new_start > 0:
                                        if DEBUG_PRINT: print("=== Case 1.1 - ", end='')
                                        # Shift to the last turn
                                        istart += new_start
                                        add_prefix = no_prefix
                                        assert text_ids[istart] == tag_token_prefix
                                    else:
                                        if DEBUG_PRINT: print("=== Case 1.2 - ", end='')
                                        # We stay in the same big turn, or it's the last turn
                                        istart += effective_block_size
                                        add_prefix = selec[:candidate_end+1]
                                        if bos and add_prefix[0] != tokenizer.bos_id:
                                            # Add BOS
                                            add_prefix = torch.cat([bos_seq, add_prefix])
                                        if istart < len(text_ids):
                                            # Avoid to cut in the middle of a word
                                            while " " not in tokenizer.decode(text_ids[istart-1:istart+1]):
                                                istart -= 1
                                else:
                                    if DEBUG_PRINT: print("=== Case 2 - ", end='')
                                    # We are in the middle of a big conversation with the same speaker,
                                    # or end of conversation
                                    istart += effective_block_size
                                    if istart < len(text_ids):
                                        # Avoid to cut in the middle of a word
                                        while " " not in tokenizer.decode(text_ids[istart-1:istart+1]):
                                            istart -= 1
                                assert istart > previous_istart

                            num_cuts += 1
                        else:
                            a = np.array(text_ids, dtype=dtype)
                            if effective_block_size and padding and len(a) < effective_block_size:
                                a = np.pad(a, (0, effective_block_size - len(a)), mode="constant", constant_values=pad_id)
                            min_len = min(min_len, len(a))
                            max_len = max(max_len, len(a))
                            if effective_block_size:
                                assert len(a) == effective_block_size
                            if build_it:
                                builder.add_array(a)
                            if ivariant == 0:
                                num_segments += 1
                            num_segments_augmented += 1
                        num_convs_augmented+= 1
                    num_convs += 1

                except Exception as err:
                    raise RuntimeError(f"Error while processing {text[:1000]}") from err


            if build_it:
                builder.write_reminder()

        print(f"* {num_cuts}/{num_convs_augmented} text cutted in several chunks")
        print(f"* min-max length: {min_len} - {max_len}")

        info = {
            "dataset": set_name,
            "conversations_check": num_convs,
            "conversations_augmented": num_convs_augmented,
            f"segments_{effective_block_size}": num_segments,
            f"segments_augmented_{effective_block_size}": num_segments_augmented,
        }
        print(json.dumps(info, indent=4))

        if update_metadata:

            with filelock.FileLock(metadata_filename_extra + ".lock", timeout=5):
                if os.path.isfile(metadata_filename_extra):
                    with open(metadata_filename_extra) as f:
                        metadata = list(csv.DictReader(f))
                else:
                    metadata = []
                metadata_dict = {row["dataset"]: row for row in metadata}
                metadata_dict[set_name] = metadata_dict.get(set_name, {}) | info
                metadata = list(metadata_dict.values())
                fieldnames = list(metadata_dict[set_name].keys())
                with open(metadata_filename_extra, "w", newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(sorted(metadata, key=lambda x: x["dataset"]))

def common_prefix(lists):
    i = 0
    min_length = min([len(l) for l in lists])
    while i < min_length and all([l[i] == lists[0][i] for l in lists[1:]]):
        i += 1
    return lists[0][:i]

def common_suffix(lists):
    return common_prefix([l[::-1] for l in lists])[::-1]


def prepare(
    source_path: Path = Path("data/source_data_folder"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    destination_path: Path = Path("data/prepared_data_folder"),
    multiple_of: int = 8,
    padding: bool = True,
    group_datasets_by_genre: bool = True,
    update_weights: bool = False,
    update_metadata: bool = False,
) -> None:
    """Prepare the "Claire" dataset. We assume tokenizer has been trained."""

    config_file = checkpoint_dir / "lit_config.json"
    config = Config.from_json(config_file)

    destination_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, destination_path / "lit_config.json")

    # Copy code used to produce the data
    (destination_path / "src").mkdir(parents=True, exist_ok=True)
    for file in __file__, "data/claire_metadata.csv":
        shutil.copy2(this_folder / file, destination_path / "src" / os.path.basename(file))
    for folder in "utils", "lit_gpt/lit_gpt", :
        shutil.copytree(this_folder / folder, destination_path / "src" / folder,
            ignore=lambda x, y: ["__pycache__"], dirs_exist_ok=True)

    effective_block_size = config.block_size + 1
    tokenizer_config_file = checkpoint_dir / "tokenizer_config.json"
    if tokenizer_config_file.is_file():

        shutil.copy2(tokenizer_config_file, destination_path / "tokenizer_config.json")
        # tokenizer_config = json.load(open(tokenizer_config_file))
        # assert config.block_size == tokenizer_config["model_max_length"]

    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        effective_block_size=effective_block_size,
        multiple_of=multiple_of,
        group_datasets_by_genre=group_datasets_by_genre,
        padding=padding,
        update_weights=update_weights,
        update_metadata=update_metadata,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
