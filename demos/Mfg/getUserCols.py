import json

def getCols(config):
    '''
    JSON file has each training file column and whether to include in the analysis
    '''
    with open(config["JSONloc"] + "colSelection.json", "r") as f:
        data = json.load(f)
    keep = []
    for d in data:
        if d["include"] == "Y":
            keep.append(d["column"])
    # Always need the label
    keep.append(config["labelColumn"])
    return keep