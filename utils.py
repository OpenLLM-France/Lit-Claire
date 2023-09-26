import csv
import os
import re

wd = os.path.dirname(os.path.realpath(__file__))

METADATA_ROWS = csv.DictReader(open(os.path.join(wd, "data", "claire_weights.csv")))
METADATA_DICT = {}
for row in METADATA_ROWS:
    METADATA_DICT[row["dataset"]] = row

def get_metadata(path):
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
