import numpy  as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def forecastRF(dataDict, trees, depth, features):
    regr = RandomForestRegressor(n_estimators = trees, max_depth=depth, max_features=features)
    
    trainX = dataDict["trainX"]
    testX  = dataDict["testX"]
    trainY = dataDict["trainY"]
    
    regr.fit(trainX, trainY)
    return regr.predict(testX)