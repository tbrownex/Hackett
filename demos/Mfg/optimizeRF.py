import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getModelParms import getParms
from evaluate      import evaluate

def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "RFmodel", 'wb'))

def loadParms(p):
    params = {'n_estimators': p[0],\
              'min_samples_split': p[1],\
              'max_depth': p[2],\
              "min_samples_leaf": p[3],\
              "max_features": p[4]}
    return params

def formatPreds(dataDict, preds):
    ''' Prepare the data to be evaluated '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["RF"]     = preds
    d["unit"]   = dataDict["testUnits"]
    df = pd.DataFrame(d)
    df.set_index("unit", inplace=True)
    return df

def process(dataDict, parms, config):    
    bestRMSE = np.inf
    bestPreds = None
    for p in parms:
        params = loadParms(p)
        regr  = RandomForestRegressor(**params)
        regr.fit(dataDict["trainX"], dataDict["trainY"])
        preds = regr.predict(dataDict["testX"])
        df    = formatPreds(dataDict, preds)        
        errors = evaluate(df)
        #mape = errors["RF"]["mape"]
        rmse = errors["RF"]["rmse"]
        if rmse < bestRMSE:
            bestRMSE = rmse
            bestPreds = preds
            saveModel(regr, config)
    return bestRMSE, bestPreds
    
def buildRF(dataDict, config):    
    parms = getParms("RF")
    
    rmse, bestPreds = process(dataDict, parms, config)
    return rmse, bestPreds