import pandas as pd
from sklearn.model_selection import train_test_split
from genFeatures import genFeatures

def splitLabels(train, test, config):
    ''' Separate the features and labels  '''
    d = {}
    d["trainY"] = train[config["labelColumn"]]
    del train[config["labelColumn"]]
    d["trainX"] = train
    
    d["testY"] = test[config["labelColumn"]]
    del test[config["labelColumn"]]
    d["testX"] = test
    return d

def prepData(df, config):
    df = df.sample(frac=config["trainingFraction"])
    del df["unit"]
    train, test = train_test_split(df, test_size=config["testPct"])
    return splitLabels(train, test, config)