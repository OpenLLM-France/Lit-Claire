import glob
import os
import sys
import re
from tqdm import tqdm
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # If python paths are not set, and that we want to test with this file standalone
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path = [parent_path + "/lit_gpt"] + sys.path
    sys.path.append(parent_path)

from utils.metadata import get_metadata, get_filename_prefix

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset

def create_dataloaders(
    batch_size=32,
    path="/gpfsscratch/rech/qgz/commun/preprocessed_data/Claire/lit-gpt/padded/tiiuae--falcon-7b/",
    prefixes=None,
    block_size=2048, # TODO: find automatically (this holds for Falcon-7b)
    shuffle=True,
    num_processes=1,
    process_rank=0,
    seed=51,
    verbose=True,
    try_small=False,
    return_details=False,
):
    pass


def create_dataloader(
    batch_size=32,
    path="/gpfsscratch/rech/qgz/commun/preprocessed_data/Claire/lit-gpt/padded/tiiuae--falcon-7b/",
    prefixes=None,
    block_size=2048, # TODO: find automatically (this holds for Falcon-7b)
    shuffle=True,
    num_processes=1,
    process_rank=0,
    seed=51,
    verbose=True,
    try_small=False,
    return_details=False,
):
    if prefixes is None:
        if try_small:
            prefixes = ["EN--ASR-ETELECSC", "FR--ParisStories", "FR--UBS"]
        else:
            prefixes = list(set(
                [get_filename_prefix(filename) for filename in os.listdir(path) \
                if os.path.isfile(os.path.join(path, filename)) and filename.endswith(".bin")]
            ))

    datasets = []
    weights = []
    pseudos = []
    for prefix in prefixes:
        filenames = glob.glob(os.path.join(path, f"{prefix}*"))
        assert len(filenames) > 0, f"No files found for prefix {prefix} in {path}"

        # Get the weight
        try:
            metadata = get_metadata(filenames[0])
        except Exception as e:
            raise RuntimeError(f"Error while getting metadata for {filenames[0]}") from e
        weights.append(metadata["sampling_rate"])

        # Only for printing information
        pseudos.append(metadata["dataset"])

        dataset = PackedDataset(
            filenames,
            n_chunks=len(filenames),
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=num_processes,
            process_rank=process_rank,
        )
        datasets.append(dataset)

    # Normalize the weights
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    if verbose:
        for w, p in sorted(zip(weights, pseudos)):
            print(f"* {p:30}: {w*100} %")

    # Combine datasets
    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    combined_dataset = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    if return_details:
        return {
            "combined_dataset": combined_dataset,
            "pseudos": pseudos,
            "datasets": datasets,
        }
    return combined_dataset


if __name__ == "__main__":

    """Test the dataset."""

    from lit_gpt.tokenizer import Tokenizer
    from pathlib import Path
    import torch
    import random
    import time

    batch_size = 4
    max_batches = 1000
    shuffle = False
    seed = random.randint(1, 1000)
    try_small = True

    checkpoint_dir = "/gpfswork/rech/qgz/commun/Claire/checkpoints/tiiuae/falcon-7b"
    tokenizer = Tokenizer(Path(checkpoint_dir))
    block_size = 2048

    import hashlib
    import pickle
    def hashmd5(obj):
        if isinstance(obj, torch.Tensor):
            return hashmd5(obj.tolist())
        return hashlib.md5(pickle.dumps(obj)).hexdigest()

    # Get the dataset
    tic = time.time()
    details = create_dataloader(
        try_small=try_small,
        return_details=True,
        block_size=block_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    print(f"Intantiation time: {time.time() - tic} seconds")

    combined_dataset = details["combined_dataset"]
    datasets = details["datasets"]
    pseudos = details["pseudos"]

    # Collect all the data from each dataset naively
    all_datas = []
    for i, d in enumerate(datasets):
        all_datas.append([])
        for s in d:
            all_datas[-1].append(hashmd5(s))
    NULL_DATA = hashmd5(torch.tensor([tokenizer.eos_id] * block_size, dtype=torch.int64))

    # Sample the combined dataset
    sample_indices = []
    stats = {}
    tic = time.time()
    useless_computation_time = 0
    for ibatch, batch in tqdm(enumerate(combined_dataset), total=max_batches):
        if ibatch == 0:
            toc = time.time()
        subtic = time.time()
        if ibatch >= max_batches:
            break
        new_batch = []
        for sample in batch:
            sample = hashmd5(sample)
            which_dataset = None
            which_index = None
            for idataset, d in enumerate(all_datas):
                if sample == NULL_DATA:
                    which_dataset = -1
                    which_index = -1
                    break
                for idata, x in enumerate(d):
                    if sample == x:
                        which_dataset = idataset
                        which_index = idata
                        break
                if which_dataset is not None:
                    break
            assert which_dataset is not None
            assert which_index is not None
            new_batch.append((which_dataset, which_index) if which_dataset >= 0 else None)
            stats[which_dataset] = stats.get(which_dataset, 0) + 1
        if len(new_batch) != batch_size:
            print(f"WARNING: Batch size is {len(new_batch)} instead of {batch_size}")
        sample_indices += new_batch
        useless_computation_time += time.time() - subtic
    print(f"Sampling time (first)  : {toc - tic} seconds")
    print(f"Sampling time (average): {(time.time() - tic - useless_computation_time)/(min(ibatch+1, max_batches)-1)} seconds")

    if len(new_batch) != batch_size:
        print(f"WARNING: Batch size is {len(new_batch)} instead of {batch_size}")
    if ibatch != max_batches:
        print(f"WARNING: Number of batches is {ibatch} instead of {max_batches}")

    # print(sample_indices)

    total = sum(stats.values())
    total_null = stats.get(-1, 0)
    if -1 in stats:
        stats.pop(-1)
    print(f"{total_null*100/total:.2f} % of samples are just padding")
    for i in sorted(stats.keys(), key=lambda x: stats[x]):
        print(f"{stats.get(i,0)*100/(total - total_null):.2f} % of samples are from dataset {pseudos[i]}")

    not_null_indices = [x for x in sample_indices if x is not None]
    num_uniques = len(set(not_null_indices))
    if len(not_null_indices) != num_uniques:
        print(f"WARNING: There are {len(not_null_indices) - num_uniques}/{len(not_null_indices)} duplicates")
