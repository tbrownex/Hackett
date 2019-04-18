import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getModelParms import getParms
from calcRMSE      import calcRMSE

''' def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "RFmodel", 'wb'))'''

def loadParms(p):
    params = {'n_estimators': p[0],\
              'min_samples_split': p[1],\
              'max_depth': p[2],\
              "min_samples_leaf": p[3],\
              "max_features": p[4]}
    return params

def process(panelDict, parms, config):    
    bestRMSE = np.inf
    bestPreds = None
    for p in parms:
        params = loadParms(p)
        regr  = RandomForestRegressor(**params)
        regr.fit(panelDict["trainX"], panelDict["trainY"])
        preds = regr.predict(panelDict["testX"])
        rmse = calcRMSE(panelDict["testY"], preds)
        if rmse < bestRMSE:
            bestRMSE = rmse
            bestPreds = preds
    return bestPreds
    
def buildRF(panelDict, config):
    parms = getParms("RF")
    
    predsDF = process(panelDict, parms, config)
    return predsDF