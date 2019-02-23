''' Prepare the data for modeling:
    - Shuffle the Training data
    - Remove any non-features
    - Identify any static columns (single value) and remove them
    - Split the features from the labels
    - Normalize the data
    - Remove outliers
    '''
__author__ = "Tom Browne"

#from normalizeData import normalize
#from analyzeCols import analyzeCols
#from removeOutliers import removeOutliers
#from genFeatures import genFeatures
from splitData  import splitData
from splitLabel import splitLabel

keep = [1,24,28,63,65,66,70]    # These are the only somewhat correlated columns with Volume/Revenue
labelCols = ["Valmont_Net_Sales_Pounds", "Valmont_Net_Sales"]

def renameCols(df):
    cols = df.columns
    
    for col in cols:
        x = col[4:]
        n = x.find("_")
        key = x[:n]
        key = "MEI_" + key
        df.rename({col:key},axis="columns", inplace=True)
    return df

def removeCols(dataDict):
    cols   = dateDict["train"].columns
    remove = analyzeCols(dateDict["train"])
    keep = [col for col in cols if col not in remove]
    dateDict["train"] = dateDict["train"][keep]
    cols   = dateDict["test"].columns             # remember test has an extra column over train (unit)
    keep = [col for col in cols if col not in remove]
    dateDict["test"]  = dateDict["test"][keep]
    return dataDict

def createLabel(df):
    df["label"] = df[config["labelColumn"]]
    del df[config["labelColumn"]]
    return df
    
def preProcess(df, config, args):
    df = renameCols(df)
    df = df.sample(frac=1).reset_index(drop=True)     # for shuffling the data
    df = df[keep]
    dataDict = splitData(df, config)
    dataDict = splitLabel(dataDict, config)
    '''dataDict  = normalize(dataDict, "Std")
    if args.Outliers == "Y":
        dataDict = removeOutliers(dataDict)'''
    return dataDict