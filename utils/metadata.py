import csv
import os
import re

wd = os.path.dirname(os.path.realpath(__file__))

metadata_filename = os.path.join(os.path.dirname(wd), "data", "claire_metadata.csv")
assert os.path.isfile(metadata_filename), f"Metadata file {metadata_filename} not found."

metadata_filename_extra = os.path.join(os.path.dirname(wd), "data", "claire_metadata_extra.csv")

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

if os.path.isfile(metadata_filename_extra):
    with open(metadata_filename_extra, "r") as csvfile:
        metadata_rows = csv.DictReader(csvfile)
        for row in metadata_rows:
            METADATA_DICT[row["dataset"]].update(format_dict_values(row))

# Add sampling weights
def get_scaled_num_samples(metadata):
    # num_samples = metadata.get("segments_augmented_2048", 0)
    num_samples = metadata["words"]
    # augmentation_factor = metadata.get("segments_augmented_2048", 1) / metadata.get("segments_2048", 1)
    # num_samples *= augmentation_factor
    if not metadata["spontaneous"]:
        num_samples /= 5
    return num_samples
scale_per_languages = {
    "fr": 0.25,
    "en": 0.75,
}
num_samples_per_language = {}
for dataset, metadata in METADATA_DICT.items():
    num_samples_per_language[metadata["language"]] = num_samples_per_language.get(metadata["language"], 0) + get_scaled_num_samples(metadata)
num_languages = len(num_samples_per_language)
for dataset, metadata in METADATA_DICT.items():
    metadata["sampling_rate"] = 100. * get_scaled_num_samples(metadata) * scale_per_languages[metadata["language"]] / (num_samples_per_language[metadata["language"]])

def get_metadata(path):
    """Get metadata from a path."""
    if not os.path.isdir(path):
        filename = os.path.basename(os.path.realpath(path))
        filename = re.sub(r"(\*)?(_[\d]+)?(\.[a-z]+)?","", filename)
        filename = filename.replace("--", "/")
        if filename in METADATA_DICT:
            return METADATA_DICT[filename]
        return get_metadata(os.path.dirname(os.path.realpath(path)))
    assert os.path.isdir(path), f"Path {path} does not exist."
    set_name = os.path.basename(path)
    if set_name in METADATA_DICT:
        return METADATA_DICT[set_name]
    set_name = os.path.basename(os.path.dirname(path)) + "/" + set_name
    assert set_name in METADATA_DICT, f"Dataset {set_name} not found in metadata file."
    return METADATA_DICT[set_name]


if __name__ == "__main__":    

    import json
    print(json.dumps(METADATA_DICT,indent=4))
    sum_per_language = {}

    print(f"{'dataset':40} {'weights':8} {'segments(A)':10} {'segments':10} {'convs':10} {'M words':7}")
    for k, v in METADATA_DICT.items():
        print(f"{k:40} {v['sampling_rate']:8.2f} {v.get('segments_augmented_2048',0):10d} {v.get('segments_2048',0):10d} {v['conversations']:10d} {v['words']/1000000:7.1f}")
        sum_per_language[v["language"]] = sum_per_language.get(v["language"], 0) + v["sampling_rate"]
    print(sum_per_language)
