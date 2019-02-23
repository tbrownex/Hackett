''' Prepare the data for modeling:
    - Shuffle the Training data
    - Remove non-features "set" and "unit"
    - Identify any static columns (single value) and remove them
    - Split the features from the labels
    - Normalize the data
    - (optional) Remove outliers
    - No need to train/test split: there's a separate file for Test
    '''
__author__ = "Tom Browne"

from normalizeData import normalize
from analyzeCols import analyzeCols
from removeOutliers import removeOutliers
from genFeatures import genFeatures

def removeCols(train, test):
    cols   = train.columns
    remove = analyzeCols(train)
    keep = [col for col in cols if col not in remove]
    train = train[keep]
    cols   = test.columns             # remember test has an extra column over train (unit)
    keep = [col for col in cols if col not in remove]
    test  = test[keep]
    return train, test

def splitLabels(train, test, config):
    '''  Separate the features and labels  '''
    d = {}
    d["trainY"] = train[config["labelColumn"]]
    del train[config["labelColumn"]]
    d["trainX"] = train
    
    d["testY"] = test[config["labelColumn"]]
    del test[config["labelColumn"]]
    d["testX"] = test
    return d

def preProcess(train, test, config, args):
    del train["set"]
    del test["set"]
    
    train, test = removeCols(train, test)
    
    if args.genFeatures == "Y":
        print("\nGenerating features")
        df = genFeatures(train, test)

    del train["unit"]   # Keep unit on test so we can plot predictions by unit

    dataDict    = splitLabels(train, test, config)
    dataDict    = normalize(dataDict, "Std")
    if args.Outliers == "Y":
        dataDict = removeOutliers(dataDict)
    # Shuffle the training data
    train = train.sample(frac=1).reset_index(drop=True)
    return dataDict