import pandas as pd
from getConfig import getConfig
'''
Some simple edits to make sure the Config parameters are ok:
- Does the specified Label column exist?
- Are there enough rows to train and test?
'''

def editConfig():
    config = getConfig()
    df = pd.read_csv(config["dataLoc"]+config["fileName"], nrows=5)
    
    assert (config["labelColumn"] in df.columns), "Invalid Config: labelColumn missing"

    def countRows():
        with open(config["dataLoc"]+config["fileName"]) as f:
            return sum(1 for line in f)

    rowCount = countRows()
    rowsNeeded = config["trainingSize"] + config["testSize"] + config["numForecasts"]
    assert (rowCount > rowsNeeded), "Invalid Config: not enough rows"