import csv
import json
import os
import re

wd = os.path.dirname(os.path.realpath(__file__))

metadata_filename = os.path.join(os.path.dirname(wd), "data", "claire_metadata.csv")
assert os.path.isfile(metadata_filename), f"Metadata file {metadata_filename} not found."

metadata_filename_extra = os.path.join(os.path.dirname(wd), "data", "claire_metadata_extra.csv")

groups_filename = os.path.join(os.path.dirname(wd), "data", "claire_data_groups.json")

def format_dict_values(d):
    for k, v in d.items():
        if isinstance(v, str):
            if re.match(r"^\d+$", v):
                d[k] = int(v)
            elif v in ["True", "False"]:
                d[k] = True if v.lower() == "true" else False
    return d

# Read CSV
METADATA_DICT = {}
with open(metadata_filename, "r") as csvfile:
    metadata_rows = csv.DictReader(csvfile)
    for row in metadata_rows:
        METADATA_DICT[row["dataset"]] = format_dict_values(row)

# Read groups
MUST_BE_EQUAL = ["spontaneous", "text", "language", "is_dev"]
with open(groups_filename, "r") as jsonfile:
    dataset_to_group = json.load(jsonfile)
group_to_datasets = {}
for dataset, group in list(dataset_to_group.items()):
    assert group not in METADATA_DICT, f"Dataset {group} already in metadata."
    for subset in "", "/TRAIN", "/TEST":
        if dataset+subset not in METADATA_DICT:
            continue
        has_found_one = True
        group_to_datasets[group+(subset or "/TRAIN")] = group_to_datasets.get(group+subset, []) + [dataset+subset]
        dataset_to_group[dataset+subset] = group+(subset or "/TRAIN")
    assert has_found_one, f"Dataset {dataset} not found in metadata."

def accumulate_metadata_by_group(datasets, metadatas=None):
    if metadatas is None:
        metadatas = [get_metadata(dataset) for dataset in datasets]
    assert len(metadatas) == len(datasets)
    new_ones = {}
    group_to_original = {}
    for d, m in zip(datasets, metadatas):
        pseudo = get_pseudo(d)
        group = dataset_to_group.get(pseudo, pseudo)
        group_to_original[group] = group_to_original.get(group, []) + [d]
        if group not in new_ones:
            new_ones[group] = m.copy()
            new_ones[group]["dataset"] = group
        else:
            for k, v in m.items():
                if k == "dataset":
                    pass
                elif k in MUST_BE_EQUAL:
                    assert new_ones[group][k] == v, f"Dataset {d} has {k}={v} but {new_ones[group][k]} expected."
                elif isinstance(v, (float, int)):
                    new_ones[group][k] += v
                else:
                    assert v in [True, False], f"Unexpected {k}={v} for {d}."
                    if new_ones[group][k] != v:
                        new_ones[group][k] = None
    groups = list(new_ones.keys())
    groups = [group_to_original[group] for group in groups]
    metadatas = list(new_ones.values())
    return groups, metadatas

# Add sampling weights
def get_scaled_num_samples(metadata):
    num_samples = metadata["words"] + metadata["turns"]
    # penalty factors
    if not metadata["spontaneous"]: # Theatre, Assemblée Nationale + All "text" datasets from English
        num_samples /= 4
    if "AssembleeNationale" in metadata["dataset"] or "MediaSum" in metadata["dataset"]: # Assemblée Nationale or MediaSum
        num_samples /= 4
    if "Europarl" in metadata["dataset"]: # Half the penalty for Europarl
        num_samples /= 2
    return num_samples

scale_per_languages = {
    "fr": 0.5,
    "en": 0.5,
}
num_samples_per_language = {}
for dataset, metadata in METADATA_DICT.items():
    num_samples_per_language[metadata["language"]] = num_samples_per_language.get(metadata["language"], 0) + get_scaled_num_samples(metadata)
num_languages = len(num_samples_per_language)
for dataset, metadata in METADATA_DICT.items():
    metadata["sampling_rate"] = 100. * get_scaled_num_samples(metadata) * scale_per_languages[metadata["language"]] / (num_samples_per_language[metadata["language"]])

def get_metadata(path):
    return METADATA_DICT[get_pseudo(path)].copy()

def get_pseudo(path):
    """Get metadata from a path."""
    if os.path.sep != "/":
        path = path.replace(os.path.sep, "/")
    if max(path.endswith(ext) for ext in [".txt", ".bin"]):
        filename = os.path.basename(path)
        filename = get_filename_prefix(filename)
        filename = filename.replace("--", "/")
        if filename in METADATA_DICT:
            return filename
        foldername = os.path.dirname(path)
        if filename in ["train"]:
            foldername += "/TRAIN"
        elif filename in ["test", "dev"]:
            foldername += "/TEST"
        try:
            return get_pseudo(foldername)
        except RuntimeError as e:
            raise RuntimeError(f"Could not find a correspondance for {path} in metadata file.") from e
    fields = path.rstrip("/").split("/")
    for k in range(1, 4):
        set_name = "/".join(fields[-k:])
        if set_name in METADATA_DICT:
            return set_name
    raise RuntimeError(f"Could not find a correspondance for {path} in metadata file.")

def get_filename_prefix(filename):
    return re.sub(r"(\*)?(_)?([\d]+)?(\.[a-z]+)?$", "", filename)


if __name__ == "__main__":

    import json
    print(json.dumps(METADATA_DICT,indent=4))
    sum_per_language = {}

    print(f"{'dataset':40} {'weights':8} {'segments(A)':10} {'segments':10} {'convs':10} {'M words':7}")
    for k, v in METADATA_DICT.items():
        print(f"{k:40} {v['sampling_rate']:8.2f} {v.get('segments_augmented_2048',0):10d} {v.get('segments_2048',0):10d} {v['conversations']:10d} {v['words']/1000000:7.1f}")
        sum_per_language[v["language"]] = sum_per_language.get(v["language"], 0) + v["sampling_rate"]
    print(sum_per_language)
