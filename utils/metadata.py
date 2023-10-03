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

# Add sampling weights
def get_scaled_num_samples(metadata):
    num_samples = metadata["words"]
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
    if os.path.sep != "/":
        path = path.replace(os.path.sep, "/")
    if max(path.endswith(ext) for ext in [".txt", ".bin"]):
        filename = os.path.basename(path)
        filename = get_filename_prefix(filename)
        filename = filename.replace("--", "/")
        if filename in METADATA_DICT:
            return METADATA_DICT[filename]
        foldername = os.path.dirname(path)
        if filename in ["train"]:
            foldername += "/TRAIN"
        elif filename in ["dev"]:
            foldername += "/DEV"
        try:
            return get_metadata(foldername)
        except RuntimeError as e:
            raise RuntimeError(f"Could not find a correspondance for {path} in metadata file.") from e
    fields = path.rstrip("/").split("/")
    for k in range(1, 4):
        set_name = "/".join(fields[-k:])
        if set_name in METADATA_DICT:
            return METADATA_DICT[set_name]
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
