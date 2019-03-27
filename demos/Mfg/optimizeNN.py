import pandas as pd
import numpy  as np
import tensorflow as tf
from pathlib import Path
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getModelParms import getParms
from kerasNN import runNN
from evaluate import evaluate

def saveModel(nn, config):
    tf.keras.models.save_model(model=nn, filepath=config["modelDir"]+"NNmodel.h5")
    
def createVal(d):
    # Split Training into Train and Val. They are already shuffled so just take bottom 20% for Val
    valSize = int(d["trainX"].shape[0]*.2)
    trainSize = d["trainX"].shape[0] - valSize
    d["trainX"] = d["trainX"].head(trainSize)
    d["trainY"] = d["trainY"].head(trainSize)
    d["valX"] = d["trainX"].tail(valSize)
    d["valY"] = d["trainY"].tail(valSize)
    
    return d

def loadParms(p):
    params = {'l1Size':     p[0],
              'activation': p[1],
              'batchSize':  p[2],
              'lr':         p[3],
              'std':        p[4],
              'dropout':    p[5],
              'optimizer':  p[6]}
    return params

def formatPreds(dataDict, preds):
    ''' Prepare the data to be evaluated '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["NN"]     = preds
    d["unit"]   = dataDict["testUnits"]
    df = pd.DataFrame(d)
    df.set_index("unit", inplace=True)
    return df

def process(dataDict, parms, config):    
    bestRMSE = np.inf
    bestPreds = None
    for p in parms:
        parmDict = loadParms(p)
        preds, nn = runNN(dataDict, parmDict, config)
        df       = formatPreds(dataDict, preds)
        errors   = evaluate(df)
        #mape = errors["NN"]["mape"]
        rmse = errors["NN"]["rmse"]
        if rmse < bestRMSE:
            bestRMSE  = rmse
            bestPreds = preds
            saveModel(nn, config)
    return bestRMSE, bestPreds

def buildNN(dataDict, config):
    dataDict = createVal(dataDict)
    
    parms = getParms("NN")
    
    rmse, preds = process(dataDict, parms, config)
    return rmse, preds