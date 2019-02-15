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
import logging
from getBaselinePreds  import getBaselinePreds
from getRFpreds  import getRFpreds
from getNNpreds  import getNNpreds
from getXGBpreds import getXGBpreds

def formatPredictions(predictions, cols):
    df = pd.DataFrame(predictions)
    df.columns = cols
    return df

def process(dataDict, config):
    cols = []
    # Baseline first
    predictions = getBaselinePreds(dataDict)
    cols.append("Baseline")
    
    # Random Forest
    preds = getRFpreds(dataDict, config)
    predictions = np.append(predictions, preds, axis=1)
    cols.append("RF")
    
    # Neural Network
    preds     = getNNpreds(dataDict, config)
    predictions = np.append(predictions, preds, axis=1)
    cols.append("NN")
        
    # XGBoost
    preds       = getXGBpreds(dataDict, config)
    predictions = np.append(predictions, preds, axis=1)
    cols.append("XGB")
    
    predictions = formatPredictions(predictions, cols)
    
    return predictions