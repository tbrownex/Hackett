import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getModelParms import getParms
from evaluate      import evaluate

def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "XGBmodel", 'wb'))

def loadParms(p):
    params = {"booster": "gbtree",\
              'n_estimators': p[0],\
              'learning_rate': p[1],\
              'max_depth': p[2],\
              "min_child_weight": p[3],\
              "colsample_bytree": p[4],\
              "subsample": p[5],\
              'loss': 'ls',\
              "gamma": p[6]}
    return params

def formatPreds(dataDict, preds):
    ''' Prepare the data to be evaluated '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["XGB"]     = preds
    d["unit"]   = dataDict["testUnits"]
    df = pd.DataFrame(d)
    df.set_index("unit", inplace=True)
    return df

def process(dataDict, parms, config):    
    bestRMSE = np.inf
    bestPreds = None
    for p in parms:
        params = loadParms(p)        
        regr = xgb.XGBRegressor(**params)
        regr.fit(dataDict["trainX"], dataDict["trainY"])
        preds = regr.predict(dataDict["testX"])
        df    = formatPreds(dataDict, preds)      
        errors = evaluate(df)
        #mape = errors["XGB"]["mape"]
        rmse = errors["XGB"]["rmse"]
        if rmse < bestRMSE:
            bestRMSE = rmse
            bestPreds = preds
            saveModel(regr, config)
    return bestRMSE, bestPreds

def buildXGB(dataDict, config):    
    parms = getParms("XGB")
    
    rmse, bestPreds = process(dataDict, parms, config)
    return rmse, bestPreds