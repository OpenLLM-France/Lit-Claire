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

# TODO: Add sampling weights

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