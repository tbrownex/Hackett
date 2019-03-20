import json

def getCols(config):
    '''
    JSON file has each training file column and whether to include in the analysis
    Each demo type, e.g. "Mfg" or "HR", has its own set of columns, so the key to the file
    is the demo type
    '''
    with open(config["JSONloc"] + "colSelection.json", "r") as f:
        data = json.load(f)
    keep = []
    for d in data:
        if d["include"] == "Y":
            keep.append(d["column"])
    return keep