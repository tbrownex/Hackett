''' Prepare the data for modeling:
    - Remove the Grade/Score
    - Identify any static columns (single value) and remove them
    - Scale the data (min-max)
    - Create a label, the Score column
    - Create Train/Val/Test sets
    - Split the features from the labels
    - (optional) Remove outliers
    '''
__author__ = "Tom Browne"

from analyzeCols import analyzeCols
from splitData   import splitData
from splitLabel  import splitLabel
from scaler      import scaler
from createLabel import createLabel
#from removeOutliers import removeOutliers

def removeCols(df):
    cols   = df.columns
    remove = analyzeCols(df)
    keep = [col for col in cols if col not in remove]
    df = df[keep]
    return df

def preProcess(df, config, args):    
    df       = removeCols(df)
    df       = scaler(df)
    df       = createLabel(df)
    dataDict = splitData(df, config["valPct"], config["testPct"])
    # Split the features and labels
    dataDict = splitLabel(dataDict, config)
    #if args.Outliers == "Y":
    #    dataDict = removeOutliers(dataDict)
    return dataDict