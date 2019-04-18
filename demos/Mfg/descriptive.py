import json
import pandas as pd

from getConfig   import getConfig
from getData     import getData
from getUserCols import getCols
from analyzeCols import analyzeCols
from getHighCorr import getHighCorr

def prepData(df, config):
    # For demo purposes user won't select the Set, just use set 1
    df = df.loc[df["set"]==1]
    # See what columns, if any, have been de-selected
    keep = getCols(config)
    df = df[keep]
    # Don't need descriptives of these columns (in case user didn't de-select them)
    try:
        df.drop(columns=["set", "unit", "cycle"], inplace=True)
    except:
        pass
    return df

def getNulls(df):
    N = df.isnull().sum()
    N =N[N>0]
    l = []
    for col, count in N.iteritems():
        d = {}
        d["name"] = col
        d["count"] = count
        l.append(d)
    return l

def getMeans(df):
    l = []
    for col, m in df.mean().iteritems():
        d = {}
        d["name"] = col
        d["value"]  = round(m,2)
        l.append(d)
    return l

def getMedians(df):
    l = []
    for col, m in df.median().iteritems():
        d = {}
        d["name"] = col
        d["value"]  = round(m,2)
        l.append(d)
    return l

def getStdDevs(df):
    l = []
    for col, m in df.std().iteritems():
        d = {}
        d["name"] = col
        d["value"]  = round(m,2)
        l.append(d)
    return l

def getCategoricals(df):
    ''' These are columns which may need to "one-hot encoded". But we have to limit how many different
    categories to encode. '''
    low = 11
    high = 30
    l = []
    for col in df.columns:
        count = len(df[col].unique())
        if count < low:
            d = {}
            d["name"] = col
            d["count"] = count
            d["error"] = 0
            l.append(d)
        elif count < high:
            d = {}
            d["name"] = col
            d["count"] = count
            d["error"] = 1
            l.append(d)
    return l

def getStatic(df):
    ''' Identify single-value columns '''
    l = []
    static = analyzeCols(df)
    for col in static:
        d = {}
        d["name"] = col
        l.append(d)
    return l

def getAlphas(df):
    ''' Show the alpha columns: they may need to be treated differently '''
    l = []
    for col, typ in df.dtypes.items():
        if typ == "object":
            d = {}
            d["name"] = col
            l.append(d)
    return l

def getLowRange(df):
    ''' This is to identify a few columns that are almost static but not quite '''
    types = df.dtypes
    l = []
    for col in df.columns:
        if types.loc[col] != "object":
            width = (df[col].max() - df[col].min()) / df[col].max()
            if width < .008:
                d = {}
                d["name"] = col
                l.append(d)
    return l

def getCorr(df, threshold, data):
    '''
    Get any highly correlated columns and format properly
    "getHighCorr" returns a dictionary with a tuple of col1 and col2 as keys and correlation as value
    Don't show correlations for any static columns: they're meaningless
    '''
    d = getHighCorr(df, threshold)
    l = []
    for cols, corr in d:
        col1, col2 = cols
        if col1 in data["static"] or col2 in data["static"]:
            pass
        else:
            d = {}
            d["col1"], d["col2"]  = col1, col2
            d["corr"] = round(corr,2)
            l.append(d)
    return l

if __name__ == "__main__":
    '''
    Create a JSON file containing all sorts of descriptive info of the data.
    Each key in the JSON file is a different type of metric
    '''
    config = getConfig()
    df, _  = getData(config)    # Don't run stats on Test
    df     = prepData(df, config)

    data = {}
    
    '''df.loc[50, "sensor20"] = None
    print(df.loc[50]["sensor20"])
    df["Tom"] = "Tom"'''
    
    
    data["nulls"]   = getNulls(df)
    data["means"]   = getMeans(df)
    data["medians"] = getMedians(df)
    data["stdDevs"] = getStdDevs(df)
    data["static"]  = getStatic(df)
    data["alpha"]   = getAlphas(df)
    data["categoricals"] = getCategoricals(df)
    data["lowRange"] = getLowRange(df)
    data["highCorr"] = getCorr(df, 0.85, data)

    with open(config["JSONloc"] + "stats.json", "w") as output:
        json.dump(data, output, sort_keys=True, indent=4)