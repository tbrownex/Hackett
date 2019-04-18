import pandas as pd

__author__ = "Tom Browne"

def convertDtypes(df):
    df["date"]       = pd.to_datetime(df["date"])
    df["hour"]       = df["hour"].astype("int8")
    df["population"] = df["population"].astype("float32")
    df["dow"]        = df["dow"].astype("int8")
    df["weekNum"]    = df["weekNum"].astype("int8")
    df["month"]      = df["month"].astype("int8")
    df["holiday"]    = df["holiday"].astype("int8")
    return df

def splitCategorical(df):
    ''' Categorical columns need to be "one-hot" encoded '''
    COLS = ["block", "dow", "weekNum", "month"]
    df = pd.get_dummies(df, columns=COLS)
    return df

def splitLabels(df, config):
    '''  Separate the features and labels  '''
    d = {}
    d["labels"] = df[config["labelColumn"]]
    del df[config["labelColumn"]]
    d["features"] = df
    return d

def preProcess(df, config):
    '''
    - Convert column types
    - Handle categorical column "Block"
    - Split features and labels
    '''
    df = convertDtypes(df)
    df = splitCategorical(df)
    dataDict = splitLabels(df, config)
    dataDict["labels"].index = df["panel"]
    dataDict["features"].set_index(["panel"], inplace=True)
    return dataDict