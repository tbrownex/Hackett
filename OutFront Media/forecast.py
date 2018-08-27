import numpy  as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import xgboost as xgb

def RF(dataDict, parms):
    regr = RandomForestRegressor(n_estimators = parms["numTrees"],
                                 max_depth    =parms["depth"],
                                 max_features =parms["features"])
    
    trainX = dataDict["trainX"]
    testX  = dataDict["testX"]
    trainY = dataDict["trainY"]
    
    regr.fit(trainX, trainY)
    return regr.predict(testX).astype(int)

# Input has X and Y values. STL is a univariate method, so we skip the Xs and use on Y 
def STL(dataDict):
    values = list(dataDict["trainY"])
    values = r.ts(values, frequency=24)
    
    model  = r.stl(values, s_window = 'periodic')
    
    forecast = importr('forecast')
    fcast    = r.forecast(model, h=48, level=80)
    vector   = fcast.rx2('fitted')
    preds    = np.asarray(vector)
    
    # For some unknown reason, "forecast" is going way beyond the "h" periods I specify
    # So truncate the predictions at length of TestY
    length = len(dataDict["testY"])
    preds  = preds[:length]
    
    return preds.astype(int)

def XGB(dataDict, parms):
    regr = xgb.XGBRegressor(objective ='reg:linear',\
                            colsample_bytree = parms["cbt"],\
                            learning_rate    = parms["LR"],\
                            max_depth        = parms["maxDepth"],\
                            alpha            = parms["alpha"],\
                            n_estimators     = parms["numTrees"])
    model = regr.fit(dataDict["trainX"],dataDict["trainY"])
    preds =  model.predict(dataDict["testX"])
    return preds.astype(int)