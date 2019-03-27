import json

from getConfig import getConfig
from getData import getData

if __name__ == "__main__":
    '''
    Create JSON with each column in the file and whether to include in the analysis or not (default to include)
    '''
    config = getConfig()    
    train, _ = getData(config)
    cols = list(train.columns)
    # Don't allow user to remove the label
    label = config["labelColumn"]
    cols.remove(label)
    
    # Final json will be a list of dictionary entries: one dict for each column
    l = []
    for col in cols:
        d = {}
        d["column"] = col
        d["include"] = "Y"
        l.append(d)
        
    with open(config["JSONloc"] + "colSelection.json", "w") as output:
        json.dump(l, output, sort_keys=True, indent=4)