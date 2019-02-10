import numpy as np
import pandas as pd
import tensorflow as tf

def calcDiff(preds, trainData):
    return trainData - preds

def getOutliers(diff):
    ''' An outlier is when any column has a "diff" of more than 2.0.
    The data was Z-scored so 2.0 is 2 StdDevs away from the mean.
    "outliers" holds the indeces of rows with outliers '''
    outliers = []
    for row in diff.iterrows():
        if sum(abs(row[1])>2.0) > 1:    # row[1] holds all the column values
            outliers.append(row[0])
    return outliers
        
def getPredictions(dataDict, nn):
    ''' Run the training data through the autoencoder to get predictions '''
    data  = np.array(dataDict["trainX"])
    return nn.predict(data)

def getModel():
    return tf.keras.models.load_model(filepath="autoencoder.hdf5", compile=False)

def removeOutliers(dataDict):
    '''
    Autoencoder has been optimized. Load the model; make predictions; identify outliers
    '''
    nn       = getModel()
    preds    = getPredictions(dataDict, nn)
    diff     = calcDiff(preds, dataDict["trainX"])
    outliers = getOutliers(diff)
    count = len(outliers)
    print("\nremoving {} outliers".format(count))
    dataDict["trainX"].drop(outliers, inplace=True)
    dataDict["trainY"].drop(outliers, inplace=True)
    return dataDict