import os
import pandas as pd
import json

from getConfig import getConfig

def checkPath(config, errors):
    ''' Confirm the data location specified exists '''
    exists = os.path.isdir(config["dataLoc"])
    if not exists:
        d = {}
        d["error"] = "Data location not found"
        errors.append(d)
    return errors

def checkJSON(config, errors):
    ''' Confirm the JSON location specified exists '''
    exists = os.path.isdir(config["JSONloc"])
    if not exists:
        d = {}
        d["error"] = "JSON location not found"
        errors.append(d)
    return errors

def checkLog(config, errors):
    ''' Confirm the log location specified exists '''
    exists = os.path.isdir(config["logLoc"])
    if not exists:
        d = {}
        d["error"] = "Log location not found"
        errors.append(d)
    return errors

def checkFile(config, errors):
    ''' Confirm the training file exists in the data location '''
    df = None
    try:
        df = pd.read_csv(config["dataLoc"]+config["fileName"], nrows=5)
    except:
        d = {}
        d["error"] = "File not found"
        errors.append(d)
    return df, errors

def checkLabel(df, config, errors):
    ''' Confirm the Label column exists in the file '''
    label = config["labelColumn"]
    if label not in df.columns:
        d = {}
        d["error"] = "Label column not found"
        errors.append(d)
    return errors
    
if __name__ == "__main__":
    config = getConfig()
    errors = []
    errors = checkPath(config, errors)
    if len(errors) == 0:
        df, errors = checkFile(config, errors)
        if df is not None:
            found = checkLabel(df, config, errors)
    errors = checkJSON(config, errors)
    errors = checkLog(config, errors)
    
    with open(config["JSONloc"] + "configErrors.json", "w") as output:
        json.dump(errors, output, indent=4)