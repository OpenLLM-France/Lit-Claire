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
    sys.path.append(parent_path)
    sys.path.append(parent_path + "/lit-gpt")

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
    split_validation_in_subsets=False,
    seed=51,
    verbose=1,
    try_small=False,
    return_details=False,
    enable_validation=True,
    enable_train=True,
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

    prefixes_dev = [p for p in all_prefixes if ("--TEST" in p or "--DEV" in p)]
    prefixes_train = [p for p in all_prefixes if p not in prefixes_dev]

    if try_small:
        selection = ["ACSYNT", "SUMM-RE", "FreD", "OFROM", "Meetings", "Interviews", "Politics"]
        prefixes_train = [p for p in prefixes_train if any([s in p for s in selection])]
        prefixes_dev = [p for p in prefixes_dev if any([s in p for s in selection])]
        if len(prefixes_dev) == 0:
            prefixes_dev = prefixes_train

    if enable_train:
        assert len(prefixes_train) > 0, f"No train set found in {path}"
    if enable_validation:
        assert len(prefixes_dev) > 0, f"No dev set found in {path}"
    assert enable_train or enable_validation

    kwargs = dict(
        path=path,
        batch_size=batch_size,
        effective_block_size=effective_block_size,
        num_processes=num_processes,
        process_rank=process_rank,
        verbose=verbose,
        return_details=return_details,
    )

    no_output = (None, {"epoch_size": 0}) if return_details else None

    train = create_dataloader(
        prefixes=prefixes_train,
        shuffle=shuffle, use_weights=True,
        wrap=wrap_train,
        seed=seed,
        **kwargs) \
        if enable_train else no_output

    valid = create_dataloader(
        prefixes=prefixes_dev,
        shuffle=shuffle, use_weights=False,
        wrap=wrap_validation,
        max_samples=max_validation_samples,
        split_in_subsets=split_validation_in_subsets,
        seed=1337,
        **kwargs) \
        if enable_validation else no_output

    return (train, valid)

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
    verbose=1,
    return_details=False,
    use_weights=True,
    split_in_subsets=False,
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
            filenames = sorted(glob.glob(os.path.join(path, f"{prefix}*.bin")))
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

        assert len(filenames) > 0
        # assert len(filenames) % num_processes == 0, \
        #     
        if len(filenames) % num_processes != 0:
            if len(filenames) > num_processes:
                raise RuntimeError(f"Number of files ({len(filenames)}) must be a multiple of the number of processes ({num_processes}) : not fullfilled for {prefix}")
            print(f"WARNING: Number of files ({len(filenames)}) not a multiple of the number of processes ({num_processes})  for {prefix}. Duplicating files to reach a multiple.")
            i = -1
            while len(filenames) % num_processes != 0:
                i += 1
                filenames.append(filenames[i % len(filenames)])
                # Note: metadata["num_files"] will be updated later

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
            "num_batches": math.ceil(num_samples / batch_size),
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
        keys = ["conversations", "turns", "words", "num_samples", "num_batches"]
        total = {}
        for what in keys:
            total[what] = sum([metadata[what] for metadata in metadatas])
        print(f"Dataset composition: {total['conversations']} conversations, {total['turns']} turns, {total['words']} words, {total_samples} samples (of length {effective_block_size}), {total['num_batches']} batches (of {batch_size}):")
        if verbose < 2:
            keys.remove("num_batches")
            keys.remove("turns")
        for w, metadata in sorted(zip(weights, metadatas), key=lambda x: (x[0], x[1]["dataset"])):
            detail_string = ""
            for what in keys:
                ratio = metadata[what] / total[what]
                what_short = what.replace("num_", "").replace("conversations", "convs")
                detail_string += f" {format_number(metadata[what])} {what_short} ({ratio*100:5.2f} %)"
            if use_weights:
                detail_string += f" -- weights = {w*100:5.2f} %"
            if verbose > 1:
                detail_string += f" -- {metadata['num_files']} files"
                detail_string += f" / {metadata['n_chunks']} chunks"
                detail_string += f" / {metadata['num_processes']} processes"
            print(f"* {metadata['dataset']:30}:{detail_string}")

    # Cut data if higher than max_samples
    if max_samples and (max_samples < total_samples_with_padding):
        assert not use_weights, "Cannot use weights and max_samples at the same time"

        # We'll take the first samples of each dataset until we reach max_samples
        max_samples_per_dataset = max_samples / len(datasets)

        new_prefixes = []
        for prefix, metadata in zip(prefixes, metadatas):
            filenames = sorted(glob.glob(os.path.join(path, f"{prefix}*.bin")))
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
            split_in_subsets=split_in_subsets,
            use_weights=use_weights,
            use_progress_bar=use_progress_bar,
        )

    if split_in_subsets:
        assert not use_weights, "Cannot use weights and split_in_subsets at the same time"
        datasets = [DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True) for dataset in datasets]
        if return_details:
            return datasets, [{
                "name": metadata["dataset"],
                "metadata": metadata,
                "epoch_size": metadata["num_samples_rounded"],
            } for metadata in metadatas]
        else:
            return datasets

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

def format_number(n, _inner_call=False):
    """ print a number approximated on 4 characters """
    if isinstance(n, float):
        if n < 9.5: # 10
            return f"{n:.1f}" if _inner_call else f"{n:.2f}"
        if n < 99.5: # 100
            return f"{round(n):3}" if _inner_call else f"{n:.1f}"
        return f"{round(n):3}" if _inner_call else f"{int(n):4}"
    if n < 1000:
        return f"{int(n):4}"
    if n < 995000:
        return f"{format_number(n/1000., True)}k"
    return f"{format_number(n/1000000., True)}M"

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
        if False: # no progress bar num_samples:
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
    parser = argparse.ArgumentParser("Test dataset iterator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, default="/gpfsscratch/rech/qgz/commun/preprocessed_data/Claire/lit-gpt/padded_8_grouped/tiiuae/falcon-7b/", nargs="?")
    parser.add_argument("checkpoint_dir", type=str, default="/gpfswork/rech/qgz/commun/Claire/checkpoints/tiiuae/falcon-7b", nargs="?")
    parser.add_argument("--devices", type=int, default=1, help= "Number of devices")
    parser.add_argument("--batch_size", type=int, default=12, help= "Batch size")
    parser.add_argument("--language", default=None, help="Filter by language")
    parser.add_argument("--seed", type=int, default=random.randint(1, 1000), help="Use 0 to disable shuffling")
    parser.add_argument("--try_small", default=False, action="store_true", help="Use dataset subsampling for quick tests")
    parser.add_argument("--split_validation_in_subsets", default=False, action="store_true", help="Split validation into subsets")
    parser.add_argument("--max_validation_samples", default=None, type= int, help="Maximum number of validation samples")
    # Options when iterating
    iter_parser = parser.add_argument_group("Iterating options")
    iter_parser.add_argument("--max_train_iters", type=int, default=0, help="Max. number of training batches to iterate over")
    iter_parser.add_argument("--max_valid_iters", type=int, default=0, help="Max. number of validation batches to iterate over")
    iter_parser.add_argument("--wrap_validation", default=False, action="store_true")
    iter_parser.add_argument("--inspect", default=False, action="store_true", help="Inspect from which dataset comes each sample")
    iter_parser.add_argument("--short_samples", default=False, action="store_true", help="Show shortened samples")
    iter_parser.add_argument("--filter_samples", type=int, nargs="*", help="Index of dataset from which to show samples (0, 1, ...)")
    iter_parser.add_argument("-o", "--output", help="output folder", default=None)
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
    max_validation_samples=args.max_validation_samples if args.max_validation_samples else (200 if try_small else 1e32)
    split_validation_in_subsets = args.split_validation_in_subsets

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
        num_processes=args.devices,
        max_validation_samples=max_validation_samples,
        split_validation_in_subsets=split_validation_in_subsets,
        try_small=try_small,
        return_details=True,
        batch_size=batch_size,
        wrap_validation=wrap_validation,
        shuffle=shuffle,
        seed=seed,
    )
    print(f"Intantiation time: {time.time() - tic} seconds")

    max_train_iters = int(train_details["epoch_size"] // batch_size)
    valid_size = dev_details["epoch_size"] if isinstance(dev_details, dict) else sum([d["epoch_size"] for d in dev_details])
    max_eval_iters = max(1, int(valid_size // batch_size))
    print("Train:")
    print("* epoch size:", train_details["epoch_size"])
    print("* max_train_iters for 1 epoch:", max_train_iters)
    print("Dev:")
    print("* epoch size:", valid_size)
    print("* max_eval_iters for 1 epoch:", max_eval_iters)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    for (combined_dataset, details, max_batches) in [(trainset, train_details, args.max_train_iters), (devset, dev_details, args.max_valid_iters)]:
        if not max_batches:
            continue

        is_list_of_datasets = isinstance(combined_dataset, list)

        if is_list_of_datasets:
            combined_datasets = combined_dataset
            datasets = combined_dataset
            pseudos = [m["name"] for m in details]
        else:
            combined_datasets = [combined_dataset]
            datasets = details["datasets"]
            pseudos = [m["dataset"] for m in details["metadata"]]

        do_inspect = args.inspect or (args.output and not is_list_of_datasets)

        # Collect all the data from each dataset naively
        if do_inspect:
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
        output_filenames = {}
        for idataset, combined_dataset in enumerate(combined_datasets):
            if is_list_of_datasets:
                print("================================")
                print("Samples from ", pseudos[idataset])
                if args.output:
                    output_file = open(os.path.join(args.output, f"SPLITTED_{max_batches}x{batch_size}_" + pseudos[idataset].replace("/", "_")+".txt"), "w")
            for ibatch, batch in enumerate(combined_dataset): # tqdm(enumerate(combined_dataset), total=max_batches):
                if ibatch == 0:
                    toc = time.time()
                subtic = time.time()
                if ibatch >= max_batches:
                    break
                new_batch = []
                for sample in batch:
                    if do_inspect:
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
                        assert which_dataset is not None
                        assert which_index is not None
                        new_batch.append((which_dataset, which_index) if which_dataset >= 0 else None)
                        stats[which_dataset] = stats.get(which_dataset, 0) + 1
                    if not args.inspect or (args.filter_samples and which_dataset in args.filter_samples):
                        sample_text = tokenizer.decode(sample.clamp_min(0)).replace("\n", "\\n")
                        short_sample_text = (sample_text[:60] + " [...] " + sample_text[-60:]) if len(sample_text) > 120 else sample_text
                        if not sample_text:
                            sample_text = "***"
                        print(short_sample_text)
                        if args.output:
                            if not is_list_of_datasets:
                                assert which_dataset is not None
                                dataset_name = pseudos[which_dataset]
                                if dataset_name not in output_filenames:
                                    output_filenames[dataset_name] = open(os.path.join(args.output, f"COMBINED_{max_validation_samples}_" + dataset_name.replace("/","_")+".txt"), "w")
                                output_file = output_filenames[dataset_name]
                            print(short_sample_text if args.short_samples else sample_text, file=output_file)
                if do_inspect and len(new_batch) != batch_size:
                    print(f"WARNING: Batch size is {len(new_batch)} instead of {batch_size}")
                sample_indices += new_batch
                useless_computation_time += time.time() - subtic

        # # Print timings
        # print(f"Sampling time (first)  : {toc - tic} seconds")
        # if ibatch > 0:
        #     print(f"Sampling time (average): {(time.time() - tic - useless_computation_time)/(min(ibatch+1, max_batches)-1)} seconds")

        if ibatch != max_batches:
            print(f"WARNING: Number of batches is {ibatch} instead of {max_batches}")

        if do_inspect:

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
