import numpy as np
import pandas as pd
import tensorflow as tf
from getModels import getModels

def calcDiff(preds, dataDict):
    ''' Get the element by element mape '''
    return (dataDict["trainX"] - preds) / dataDict["trainX"]

def getOutliers(diff):
    '''
    Each row in the DF has a MAPE for each column. If a row has a certain number of high MAPEs,
    label it an outlier.
    "outliers" holds the indeces of rows with outliers
    '''
    outliers = []
    for row in diff.iterrows():
        if sum(abs(row[1]) > 2.5) > 2:    # row[1] holds all the column MAPEs
            outliers.append(row[0])
    return outliers
        
def getPredictions(nn, dataDict):
    ''' Run the training data through the autoencoder to get predictions '''
    data  = np.array(dataDict["trainX"])
    return nn.predict(data)

def loadModel(fname):
    return tf.keras.models.load_model(filepath=fname[0], compile=False)

def removeOutliers(dataDict, config):
    '''
    Autoencoder has been optimized. Load the model; make predictions; identify outliers
    '''
    fname    = getModels("AE", config)
    nn       = loadModel(fname)
    preds    = getPredictions(nn, dataDict)
    diff     = calcDiff(preds, dataDict)
    outliers = getOutliers(diff)
    count = len(outliers)
    print("\nremoving {} outliers".format(count))
    dataDict["trainX"].drop(outliers, inplace=True)
    dataDict["trainY"].drop(outliers, inplace=True)
    return dataDict