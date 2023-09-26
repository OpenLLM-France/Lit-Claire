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

from utils import get_metadata

##############################
# Text normalization functions

def collapse_whitespaces(text):
    return re.sub(r" +", " ", text).strip()

def remove_special_words(text):
    # Remove all [*] except the one at the beginning and after linebreaks
    text = re.sub(r"([^\n])\[[^\]]*\]", r"\1", text)
    return collapse_whitespaces(text)
    
def remove_punctuations(text):
    text = re.sub(r"[,\.!?…]", "", text)
    return collapse_whitespaces(text)

def to_lower_case(text):
    return text.lower()

def anonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(r"\[([^\]]+):\]", text) if x not in speakers] 
    new_speakers = [f"speaker{i+1:03d}" for i in range(len(speakers))]
    for spk, nspk in zip(speakers, new_speakers):
        text = text.replace(f"[{spk}:", f"[{nspk}:")
    return text

def has_upper_case(text):
    return bool(re.search(r"[A-Z]", text))

def has_speaker_id(text):
    return bool(re.search(r"\[[^spkeaker\d]+:\]", text))

def has_punctuation(text):
    return bool(re.search(r"[,\.!?…]", text))

def augmented_texts_generator(text):
    text = remove_special_words(text)
    yield text
    _upper = has_upper_case(text)
    _speaker = has_speaker_id(text)
    _punct = has_punctuation(text)
    if _speaker:
        text_anonym = anonymize_speakers(text)
        yield text_anonym
    if _upper:
        yield to_lower_case(text)
        if _speaker:
            yield to_lower_case(text_anonym)
    if _punct:
        text_no_punct = remove_punctuations(text)
        yield text_no_punct
        if _upper:
            yield to_lower_case(text_no_punct)
            if _speaker:
                yield remove_punctuations(to_lower_case(text_anonym))
        if _speaker:
            yield remove_punctuations(text_anonym)

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

    all_files = {}
    for root, dirs, files in os.walk(source_path, followlinks=True):
        root = os.path.realpath(root)
        for file in files:
            if re.match(filename_regex + r"$", file):
                filepath = os.path.join(root, file)
                set_name = get_metadata(filepath)["dataset"].replace("/", "--")
                all_files[filepath] = set_name

    if len(all_files) == 0:
        raise RuntimeError(
            f"No input files found at {source_path}."
        )

    for filepath, set_name in all_files.items():
        print(f"Processing {filepath} -> {destination_path}/{set_name}*")

        dataset_hf = load_dataset("text", data_files={"train": filepath}, sample_by="paragraph", streaming=True)

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

            # Text normalization
            text = remove_special_words(text)

            # Augmentation
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
