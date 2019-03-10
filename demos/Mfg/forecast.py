'''
Get predictions against the test set for each algo and also a Baseline prediction.
Baseline prediction uses the mean time to fail over the Training data.

Return a dataframe with a column for:
    - Baseline first
    - each algo
    
So the shape of the dataframe will be [number of test cases, number of algos + 1]
'''
__author__ = "Tom Browne"

import numpy  as np
import pandas as pd
from getBaselinePreds  import getBaselinePreds
from getRFpreds  import getRFpreds
from getNNpreds  import getNNpreds
from getXGBpreds import getXGBpreds

def process(dataDict, config):
    predDF = pd.DataFrame()
    
    # Baseline first
    preds              = getBaselinePreds(dataDict)
    preds              = np.reshape(preds, newshape=[-1,])
    predDF["Baseline"] = preds
    
    # Random Forest
    preds        = getRFpreds(dataDict, config)
    preds        = np.reshape(preds, newshape=[-1,])
    predDF["RF"] = preds
    
    # Neural Network
    preds        = getNNpreds(dataDict, config)
    preds        = np.reshape(preds, newshape=[-1,])
    predDF["NN"] = preds
    
    # XGBoost
    preds         = getXGBpreds(dataDict, config)
    preds         = np.reshape(preds, newshape=[-1,])
    predDF["XGB"] = preds
    
    return predDF