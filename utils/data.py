import glob
import os
import sys
import json
import math
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

def create_dataloaders(
    path,
    language=None,
    batch_size=32,
    shuffle=True,
    num_processes=1,
    process_rank=0,
    wrap_train=True,
    wrap_validation=True,
    max_validation_samples=None,
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

    if language:
        all_prefixes = [p for p in all_prefixes if p.startswith(language.upper())]

    prefixes_dev = [p for p in all_prefixes if "--DEV" in p]
    prefixes_train = [p for p in all_prefixes if p not in prefixes_dev]

    if try_small:
        selection = ["ACSYNT", "SUMM-RE", "FreD", "OFROM"]
        prefixes_train = [p for p in prefixes_train if any([s in p for s in selection])]
        prefixes_dev = [p for p in prefixes_dev if any([s in p for s in selection])]
        if len(prefixes_dev) == 0:
            prefixes_dev = prefixes_train

    assert len(prefixes_dev) > 0, "No dev set found"
    assert len(prefixes_train) > 0, "No train set found"

    kwargs = dict(
        path=path,
        batch_size=batch_size,
        effective_block_size=effective_block_size,
        num_processes=num_processes,
        process_rank=process_rank,
        seed=seed,
        verbose=verbose,
        return_details=return_details,
    )

    return (
        create_dataloader(prefixes=prefixes_train, shuffle=shuffle, wrap=wrap_train, **kwargs),
        create_dataloader(prefixes=prefixes_dev, shuffle=False, use_weights=False, wrap=wrap_validation, max_samples=max_validation_samples, **kwargs)
    )

def create_dataloader(
    path,
    effective_block_size,
    batch_size=32,
    prefixes=None,
    shuffle=True,
    num_processes=1,
    process_rank=0,
    wrap=False,
    seed=51,
    verbose=True,
    return_details=False,
    use_weights=True,
    use_progress_bar=True,
    max_samples=None,
):
    assert num_processes > 0, "num_processes must be > 0"
    if prefixes is None:
        prefixes = list(set(
            [get_filename_prefix(filename) for filename in os.listdir(path) \
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith(".bin")]
        ))

    datasets = []
    datasets_nowrap = []
    weights = []
    num_samples_per_dataset = []
    metadatas = []
    for prefix in prefixes:
        
        if isinstance(prefix, str):
            filenames = glob.glob(os.path.join(path, f"{prefix}*.bin"))
            assert len(filenames) > 0, f"No files found in {path} (for prefix: {prefix})"

            metadata_file = os.path.join(path, f"{prefix}_metadata.json")
            assert os.path.isfile(metadata_file), f"Metadata file {metadata_file} does not exist"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            expected_num_files = metadata["num_files"]
            assert expected_num_files  == len(filenames), f"Metadata file {metadata_file} specifies {expected_num_files} files but {len(filenames)} found for prefix {prefix}"
        else:
            # See below: when recursive call to reach a maximum number of samples
            assert isinstance(prefix, dict)
            filenames = prefix["filenames"]
            metadata = prefix["metadata"]

        assert len(filenames) % num_processes == 0, \
            f"Number of files ({len(filenames)}) must be a multiple of the number of processes ({num_processes}) : not fullfilled for {prefix}"

        # Get the weight
        if use_weights:
            weights.append(metadata["sampling_rate"])
        else:
            # Only for reporting
            weights.append(metadata["num_samples"])

        num_samples = metadata["num_samples_rounded"]
        num_samples_per_dataset.append(num_samples)

        n_chunks = max(1, len(filenames) // num_processes)
        num_processes_this = min(num_processes, len(filenames))
        metadata.update({
            "num_files": len(filenames),
            "n_chunks": n_chunks,
            "num_processes": num_processes_this,
        })

        kwargs = dict(
            filenames=filenames,
            n_chunks=n_chunks,
            block_size=effective_block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=num_processes_this,
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

        metadatas.append(metadata)

    # Normalize the weights
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    # Get totals
    total_samples_with_padding = sum(num_samples_per_dataset)
    total_samples = sum([metadata["num_samples"] for metadata in metadatas])

    # Print proportions and weights
    if verbose:
        print(f"Dataset composition ({total_samples} samples):")
        for w, metadata in sorted(zip(weights, metadatas), key=lambda x: (x[0], x[1]["dataset"])):
            ratio = metadata["num_samples"] / total_samples
            detail_string = ""
            if use_weights:
                detail_string += f" -- weights = {w*100:5.2f} %"
            detail_string += f" -- {metadata['num_files']} files"
            detail_string += f" / {metadata['n_chunks']} chunks"
            detail_string += f" / {metadata['num_processes']} processes"
            print(f"* {metadata['dataset']:30}: {ratio*100:5.2f} % ({metadata['num_samples']} samples){detail_string}")

    # Cut data if higher than max_samples
    if max_samples and max_samples < total_samples_with_padding:
        assert not use_weights, "Cannot use weights and max_samples at the same time"

        # We'll take the first samples of each dataset until we reach max_samples
        max_samples_per_dataset = max_samples / len(datasets)

        new_prefixes = []
        for prefix, metadata in zip(prefixes, metadatas):
            filenames = glob.glob(os.path.join(path, f"{prefix}*.bin"))
            assert len(filenames) > 0
            num_files = max(1, int((len(filenames) * max_samples_per_dataset) // metadata["num_samples"]))
            # The number of files must be a multiple of the number of processes
            num_files = min(math.ceil(num_files / num_processes) * num_processes, len(filenames))
            num_samples = num_files * metadata["num_samples_per_file"]
            new_prefixes.append({
                "filenames": filenames[:num_files],
                "metadata": metadata | {
                    "num_samples": num_samples,
                    "num_samples_rounded": num_samples,
                    "num_files": num_files,
                    "num_padded" : metadata["num_padded"] if (num_files == len(filenames)) else 0, # Only last file can contain padded-only tensors
                },
            })

        return create_dataloader(
            path,
            max_samples=None,
            prefixes=new_prefixes,
            # Rest of the arguments are the same
            effective_block_size=effective_block_size,
            batch_size=batch_size,
            shuffle=shuffle,
            num_processes=num_processes,
            process_rank=process_rank,
            wrap=wrap,
            seed=seed,
            verbose=verbose,
            return_details=return_details,
            use_weights=use_weights,
            use_progress_bar=use_progress_bar,
        )

    # Combine datasets
    if use_weights:
        combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
        epoch_size = sum([w*s for (w,s) in zip(weights, num_samples_per_dataset)])
    else:
        epoch_size = total_samples_with_padding
        combined_dataset = ConcatenatedDataset(datasets=datasets, num_samples=epoch_size if use_progress_bar else None)

    combined_dataset = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Return results
    if return_details:
        return combined_dataset, {
            "metadata": metadatas,
            "datasets": datasets_nowrap if wrap else datasets,
            "epoch_size": epoch_size,
        }
    return combined_dataset


class ConcatenatedDataset(IterableDataset):
    def __init__(self, datasets, num_samples=None):
        self._datasets = datasets
        self._num_samples = num_samples

    def __iter__(self):
        return ConcatenatedDatasetIterator(self._datasets, self._num_samples)


class ConcatenatedDatasetIterator:
    def __init__(self, datasets, num_samples=None):
        self._datasets = [iter(el) for el in datasets]
        self._idataset = 0
        # Progress bar only
        if num_samples:
            # + 1 because the progress bar is incremented at the start of each iter, including the last StopIteration
            self._pbar = tqdm(total=num_samples+1, unit="samples", desc="Serving data", initial=0)
        else:
            self._pbar = None

    def __next__(self):
        if self._pbar:
            self._pbar.update()
        try:
            return next(self._datasets[self._idataset])
        except (StopIteration, IndexError):
            self._idataset += 1
            if self._idataset >= len(self._datasets):
                # if self._pbar:
                #     self._pbar.close()
                raise StopIteration
            return next(self._datasets[self._idataset])


if __name__ == "__main__":

    """Test the dataset."""

    import random
    import argparse
    parser = argparse.ArgumentParser("Test dataset iterator")
    parser.add_argument("path", type=str, default="/gpfsscratch/rech/qgz/commun/preprocessed_data/Claire/lit-gpt/padded/tiiuae/falcon-7b/", nargs="?")
    parser.add_argument("checkpoint_dir", type=str, default="/gpfswork/rech/qgz/commun/Claire/checkpoints/tiiuae/falcon-7b", nargs="?")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--language", default=None, help="Filter by language")
    parser.add_argument("--max_batches_train", type=int, default=1000)
    parser.add_argument("--max_batches_dev", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=random.randint(1, 1000), help="Use 0 to disable shuffling")
    parser.add_argument("--try_small", default=False, action="store_true")
    parser.add_argument("--wrap_validation", default=False, action="store_true")
    parser.add_argument("--show_samples", type=int, nargs="*", help="Index of dataset from which to show samples")
    parser.add_argument("--no_iter", default=False, action="store_true", help="Do not iterate over the dataset")
    args = parser.parse_args()

    from lit_gpt.tokenizer import Tokenizer
    from pathlib import Path
    import torch
    import time

    batch_size = args.batch_size
    shuffle = bool(args.seed)
    seed = args.seed
    try_small = args.try_small
    wrap_validation = args.wrap_validation

    max_validation_samples = 200 if try_small else 4000

    checkpoint_dir = args.checkpoint_dir
    assert os.path.isdir(checkpoint_dir), f"Checkpoint dir {checkpoint_dir} does not exist"
    tokenizer = Tokenizer(Path(checkpoint_dir))

    import hashlib
    import pickle
    def hashmd5(obj):
        if isinstance(obj, torch.Tensor):
            return hashmd5(obj.tolist())
        return hashlib.md5(pickle.dumps(obj)).hexdigest()

    # Get the dataset
    tic = time.time()
    (trainset, train_details), (devset, dev_details) = create_dataloaders(
        path=args.path,
        language=args.language,
        max_validation_samples=max_validation_samples,
        try_small=try_small,
        return_details=True,
        batch_size=batch_size,
        wrap_validation=wrap_validation,
        shuffle=shuffle,
        seed=seed,
    )
    print(f"Intantiation time: {time.time() - tic} seconds")

    max_train_iters = int(train_details["epoch_size"] // batch_size)
    max_eval_iters = max(1, int(dev_details["epoch_size"] // batch_size))
    print("Train:")
    print("* epoch size:", train_details["epoch_size"])
    print("* max_train_iters for 1 epoch:", max_train_iters)
    print("Dev:")
    print("* epoch size:", dev_details["epoch_size"])
    print("* max_eval_iters for 1 epoch:", max_eval_iters)

    if args.no_iter:
        sys.exit(0)

    for (combined_dataset, details, max_batches) in [(trainset, train_details, args.max_batches_train), (devset, dev_details, args.max_batches_dev)]:

        datasets = details["datasets"]
        pseudos = [m["dataset"] for m in details["metadata"]]

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
        for ibatch, batch in tqdm(enumerate(combined_dataset), total=max_batches):
            if ibatch == 0:
                toc = time.time()
            subtic = time.time()
            if ibatch >= max_batches:
                break
            new_batch = []
            for sample in batch:
                sample_hash = hashmd5(sample)
                which_dataset = None
                which_index = None
                for idataset, d in enumerate(all_datas):
                    if sample_hash == NULL_DATA:
                        which_dataset = -1
                        which_index = -1
                        break
                    for idata, x in enumerate(d):
                        if sample_hash == x:
                            which_dataset = idataset
                            which_index = idata
                            break
                    if which_dataset is not None:
                        break
                if args.show_samples and which_dataset in args.show_samples:
                    print(f"dataset{which_dataset}", tokenizer.decode(sample)[:100])
                assert which_dataset is not None
                assert which_index is not None
                new_batch.append((which_dataset, which_index) if which_dataset >= 0 else None)
                stats[which_dataset] = stats.get(which_dataset, 0) + 1
            if len(new_batch) != batch_size:
                print(f"WARNING: Batch size is {len(new_batch)} instead of {batch_size}")
            sample_indices += new_batch
            useless_computation_time += time.time() - subtic
        print(f"Sampling time (first)  : {toc - tic} seconds")
        if ibatch > 0:
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
