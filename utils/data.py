import glob
import os
import sys
import re
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

if __name__ == "__main__":
    # If python paths are not set, and that we want to test with this file standalone
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path = [parent_path + "/lit_gpt"] + sys.path
    sys.path.append(parent_path)

from utils.metadata import get_metadata, get_filename_prefix

from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.config import Config

DEFAULT_PATH="/gpfsscratch/rech/qgz/commun/preprocessed_data/Claire/lit-gpt/padded/tiiuae--falcon-7b/"

def create_dataloaders(
    batch_size=32,
    path=DEFAULT_PATH,
    shuffle=True,
    num_processes=1,
    process_rank=0,
    seed=51,
    verbose=True,
    try_small=False,
    return_details=False,
):
    assert os.path.isdir(path), f"Path {path} does not exist"

    config_file = os.path.join(path, "lit_config.json")
    assert os.path.isfile(config_file), f"Config file {config_file} does not exist"
    config = Config.from_json(config_file)
    effective_block_size = config.block_size + 1

    all_prefixes = list(set(
        [get_filename_prefix(filename) for filename in os.listdir(path) \
        if os.path.isfile(os.path.join(path, filename)) and filename.endswith(".bin")]
    ))

    prefixes_dev = [p for p in all_prefixes if "--DEV" in p]
    prefixes_train = [p for p in all_prefixes if p not in prefixes_dev]

    if try_small:
        prefixes_train = [p for p in prefixes_train if ("EN--ASR-ETELECSC" in p or "FR--ParisStories" in p or "FR--UBS" in p)]
        prefixes_dev = [p for p in prefixes_dev if ("SUMM-RE" in p or "OFROM" in p)]
        if len(prefixes_dev) == 0:
            prefixes_dev = prefixes_train

    assert len(prefixes_dev) > 0, "No dev set found"
    assert len(prefixes_train) > 0, "No train set found"

    kwargs = dict(
        batch_size=batch_size,
        path=path,
        effective_block_size=effective_block_size,
        num_processes=num_processes,
        process_rank=process_rank,
        seed=seed,
        verbose=verbose,
        return_details=return_details,
    )

    return (
        create_dataloader(prefixes=prefixes_train, shuffle=shuffle, wrap=True, **kwargs),
        create_dataloader(prefixes=prefixes_dev, shuffle=False, use_weights=False, **kwargs)
    )

def create_dataloader(
    effective_block_size,
    batch_size=32,
    path=DEFAULT_PATH,
    prefixes=None,
    shuffle=True,
    num_processes=1,
    process_rank=0,
    wrap=False,
    seed=51,
    verbose=True,
    return_details=False,
    try_small=False,
    use_weights=True,
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
    datasets_nowrap = []
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
        if use_weights:
            weights.append(metadata["sampling_rate"])
        else:
            # Only for reporting
            weights.append(metadata["words"])

        # Only for printing information
        pseudos.append(metadata["dataset"])

        kwargs = dict(
            filenames=filenames,
            n_chunks=len(filenames),
            block_size=effective_block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=num_processes,
            process_rank=process_rank,
        )

        datasets.append(PackedDataset(
            **kwargs,
            wrap=wrap,
        ))

        datasets_nowrap.append(PackedDataset(
            **kwargs,
            wrap=False,
        ))

    # Normalize the weights
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    if verbose:
        for w, p in sorted(zip(weights, pseudos)):
            print(f"* {p:30}: {w*100} %")

    # Combine datasets
    if use_weights:
        if wrap:
            combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
            datasets = datasets_nowrap
        else:
            combined_dataset = InfiniteCombinedDataset(datasets=datasets, seed=seed, weights=weights)
    else:
        combined_dataset = ConcatenatedDataset(datasets=datasets)

    combined_dataset = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    if return_details:
        return {
            "combined_dataset": combined_dataset,
            "pseudos": pseudos,
            "datasets": datasets,
        }
    return combined_dataset


class ConcatenatedDataset(IterableDataset):
    def __init__(self, datasets):
        self._datasets = datasets

    def __iter__(self):
        return ConcatenatedDatasetIterator(self._datasets)


class ConcatenatedDatasetIterator:
    def __init__(self, datasets):
        self._datasets = [iter(el) for el in datasets]
        self._idataset = 0

    def __next__(self):
        try:
            return next(self._datasets[self._idataset])
        except (StopIteration, IndexError):
            self._idataset += 1
            if self._idataset >= len(self._datasets):
                raise StopIteration
            return next(self._datasets[self._idataset])

class InfiniteCombinedDataset(CombinedDataset):
    def __iter__(self):
        return InfiniteCombinedDatasetIterator(self._datasets, self._seed, self._weights)

class InfiniteCombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = datasets
        self._datasets_iter = [iter(el) for el in datasets]
        self._datasets_indices = list(range(len(datasets)))
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (idataset,) = self._rng.choices(self._datasets_indices, weights=self._weights, k=1)
        try:
            return next(self._datasets_iter[idataset])
        except StopIteration:
            self._datasets_iter[idataset] = iter(self._datasets[idataset])
            return next(self._datasets_iter[idataset])


if __name__ == "__main__":

    """Test the dataset."""

    import random
    import argparse
    parser = argparse.ArgumentParser("Test dataset iterator")
    parser.add_argument("path", type=str, default=DEFAULT_PATH, nargs="?")
    parser.add_argument("checkpoint_dir", type=str, default="/gpfswork/rech/qgz/commun/Claire/checkpoints/tiiuae/falcon-7b", nargs="?")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=random.randint(1, 1000))
    args = parser.parse_args()

    from lit_gpt.tokenizer import Tokenizer
    from pathlib import Path
    import torch
    import time

    batch_size = args.batch_size
    max_batches = args.max_batches
    shuffle = bool(args.seed)
    seed = args.seed
    try_small = True

    checkpoint_dir = args.checkpoint_dir
    tokenizer = Tokenizer(Path(checkpoint_dir))

    import hashlib
    import pickle
    def hashmd5(obj):
        if isinstance(obj, torch.Tensor):
            return hashmd5(obj.tolist())
        return hashlib.md5(pickle.dumps(obj)).hexdigest()

    # Get the dataset
    tic = time.time()
    train_details, dev_details = create_dataloaders(
        path=args.path,
        try_small=try_small,
        return_details=True,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    print(f"Intantiation time: {time.time() - tic} seconds")

    for details in [train_details, dev_details]:

        combined_dataset = details["combined_dataset"]
        datasets = details["datasets"]
        pseudos = details["pseudos"]

        # Collect all the data from each dataset naively
        all_datas = []
        for i, d in enumerate(datasets):
            all_datas.append([])
            for s in d:
                all_datas[-1].append(hashmd5(s))
        NULL_DATA = hashmd5(torch.tensor([tokenizer.eos_id] * 2049, dtype=torch.int64)) # 2049 hardcoded

        # Sample the combined dataset
        sample_indices = []
        stats = {}
        tic = time.time()
        useless_computation_time = 0
        for ibatch, batch in tqdm(enumerate(combined_dataset), total=args.max_batches):
            if ibatch == 0:
                toc = time.time()
            subtic = time.time()
            if ibatch >= args.max_batches:
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
        print(f"{total_null*100/total:.2f} % of samples ({total_null}) are just padding")
        for i in sorted(stats.keys(), key=lambda x: stats[x]):
            print(f"{stats.get(i,0)*100/(total - total_null):.2f} % of samples ({stats.get(i,0)}) are from dataset {pseudos[i]}")

        not_null_indices = [x for x in sample_indices if x is not None]
        num_uniques = len(set(not_null_indices))
        if len(not_null_indices) != num_uniques:
            print(f"WARNING: There are {len(not_null_indices) - num_uniques}/{len(not_null_indices)} duplicates")