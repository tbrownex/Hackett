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
from getUserCols import getCols

def removeCols(train, test):
    ''' if any column is constant (same value for all rows) remove it '''
    cols   = train.columns
    remove = analyzeCols(train)
    keep = [col for col in cols if col not in remove]
    train = train[keep]
    cols   = test.columns             # remember test has an extra column over train (unit)
    keep = [col for col in cols if col not in remove]
    test  = test[keep]
    return train, test

def userCols(dataDict, config):
    ''' the user selects what columns to analyze '''
    user = getCols(config)
    cols = dataDict["trainX"].columns
    keep = [col for col in  cols if col in user]
    dataDict["trainX"] = dataDict["trainX"][keep]
    dataDict["testX"] = dataDict["testX"][keep]
    return dataDict

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

#def preProcess(train, test, config, args):
def preProcess(train, test, config):
    '''
    - Remove constant columns
    - (optional) generate features like moving mean, std, etc.
    - Shuffle training data
    - Remove "Unit" from training data
    - Split features and labels
    - (optional) remove outliers
    '''
    train, test = removeCols(train, test)
    
    '''if args.genFeatures == "Y":
        print("\nGenerating features")
        train, test = genFeatures(train, test)'''
    
    # Shuffle the training data
    train = train.sample(frac=1).reset_index(drop=True)

    dataDict = splitLabels(train, test, config)
    
    ''' Remove Unit since its not a feature, but keep the Test units so we can recreate
    the Predictions vs Actuals by Unit '''
    del dataDict["trainX"]["unit"]
    dataDict["testUnits"] = dataDict["testX"]["unit"]
    del dataDict["testX"]["unit"]
    dataDict = userCols(dataDict, config)
    
    dataDict    = normalize(dataDict, "Std")
    '''if args.Outliers == "Y":
        dataDict = removeOutliers(dataDict, config)'''
    return dataDict