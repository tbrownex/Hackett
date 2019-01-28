''' Prepare the data for modeling:
    - Remove non-features "set" and "unit"
    - Shuffle the rows
    - No need to train/test split: there's a separate file for Test
    - Split the features from the labels
    '''
__author__ = "Tom Browne"

def splitLabels(train, test, config):
    '''
    Separate the features and labels
    '''
    d = {}
    d["trainY"] = train[config["labelColumn"]]
    del train[config["labelColumn"]]
    d["trainX"] = train
    
    d["testY"] = test[config["labelColumn"]]
    del test[config["labelColumn"]]
    d["testX"] = test
    return d

def process(train, test, config):
    del train["set"]
    del train["unit"]
    del test["set"]   # Keep unit on test so we can plot predictions by unit
    train = train.sample(frac=1).reset_index(drop=True)
    d = splitLabels(train, test, config)
    return d