''' Prepare the data for analysis:
    - Remove any duplicate columns
    - Split the features from the labels
    - Generate features
    '''
__author__ = "Tom Browne"

import numpy  as np
import pandas as pd

#from normalizeData import normalize
from genFeatures import genFeatures
from splitLabel import splitLabel

'''def renameCols(df):
    cols = df.columns
    
    for col in cols:
        x = col[4:]
        n = x.find("_")
        key = x[:n]
        key = "MEI_" + key
        df.rename({col:key},axis="columns", inplace=True)
    return df'''

def removeDups(df):
    '''
    In case we brought over the same MEI twice
    The "mean" function loses the date so save and add back in
    '''
    print("removing duplicate columns")
    dt = df["Date"]
    avg = df.mean()
    dups = avg.loc[avg.duplicated()].index.values
    if len(dups) > 0:
        for x in dups:
            print("- ", x)
    else:
        print(" - None found")
    keep = avg.loc[~avg.duplicated()]
    df = df[list(keep.index.values)]
    df["Date"] = dt
    return df

def splitData(df, config):
    trainSize = int(df.shape[0] * (1 - config["testPct"]))
    test  = df[trainSize:]
    d = {}
    d["train"] = df[:trainSize]
    d["test"]  = df[trainSize:]
    return d
    
def splitLabels(dataDict):
    ''' This data has two labels. After you "del" the labels, only MEIs remain '''
    cols = ["Date", "Volume", "Revenue"]
    d = {}
    d["trainY"] = dataDict["train"][cols]
    d["testY"]  = dataDict["test"][cols]
    
    d["trainX"] = dataDict["train"].copy()
    del d["trainX"]["Volume"]
    del d["trainX"]["Revenue"]
    d["testX"] = dataDict["test"].copy()
    del d["testX"]["Volume"]
    del d["testX"]["Revenue"]
    
    d["trainX"].set_index("Date", inplace=True)
    d["testX"].set_index("Date", inplace=True)
    d["trainY"].set_index("Date", inplace=True)
    d["testY"].set_index("Date", inplace=True)
    return d

def preProcess(df, config):
    #df = renameCols(df)
    df = removeDups(df)
    
    before = df.shape[1]
    df = genFeatures(df)
    print("\n{} features generated".format(df.shape[1] - before))
    
    before = df.shape[0]
    df = df.dropna(axis=0)
    print("{} rows removed due to NA".format(before - df.shape[0]))
    
    dataDict = splitData(df, config)    
    dataDict = splitLabels(dataDict)
    
    #dataDict  = normalize(dataDict, "Std")
    return dataDict