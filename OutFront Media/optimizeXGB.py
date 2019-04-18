import pandas as pd
import numpy  as np
import xgboost as xgb
import pickle
#import warnings
#from sklearn.exceptions import DataConversionWarning
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getModelParms import getParms
from calcRMSE      import calcRMSE

''' def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "RFmodel", 'wb'))'''

def loadParms(p):
    params = {'objective': 'reg:linear',\
              'colsample_bytree': p[0],\
              'learning_rate': p[1],\
              'max_depth': p[2],\
              'n_estimators': p[3],\
              'reg_alpha': p[4]}
    return params

def process(panelDict, parms, config):    
    bestRMSE = np.inf
    bestPreds = None
    for p in parms:
        params = loadParms(p)
        regr = xgb.XGBRegressor(**params)
        regr.fit(panelDict["trainX"], panelDict["trainY"])
        preds = regr.predict(panelDict["testX"])
        rmse = calcRMSE(panelDict["testY"], preds)
        if rmse < bestRMSE:
            bestRMSE = rmse
            bestPreds = preds
    return bestPreds
    
def buildXGB(panelDict, config):
    parms = getParms("XGB")
    
    predsDF = process(panelDict, parms, config)
    return predsDF