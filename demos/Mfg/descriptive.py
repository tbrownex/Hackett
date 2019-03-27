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
    d = {}
    for col, count in N.iteritems():
        d[col] = count
    return d

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
    d = {}
    for col in df.columns:
        count = len(df[col].unique())
        if count < low:
            d[col] = dict({"count":count, "error":0})
        elif count < high:
            d[col] = dict({"count":count, "error":1})
    return d

def getStatic(df):
    ''' Identify single-value columns '''
    return analyzeCols(df)

def getAlphas(df):
    ''' Show the alpha columns: they may need to be treated differently '''
    l = []
    for col, typ in df.dtypes.items():
        if typ == "object":
            l.append(col)
    return l

def getLowRange(df):
    ''' This is to identify a few columns that are almost static but not quite '''
    l = []
    for col in df.columns:
        width = (df[col].max() - df[col].min()) / df[col].max()
        if width < .008:
            l.append(col)
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
            tmp = {}
            tmp["col1"], tmp["col2"]  = col1, col2
            tmp["corr"] = round(corr,2)
            l.append(tmp)
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