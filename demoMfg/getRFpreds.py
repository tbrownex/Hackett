from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pickle
from getModels import getModels

def process(dataDict, config):
    '''
    Get any RF models that have been built. One should be typical (the optimized one) but more are ok
    For each model, generate predictions against the Test set
    
    "predictions" is a numpy array initialized empty, then you add a column for each model's predictions
    The shape of "predictions" is [rows of test cases, number of models]
    '''
    models = getModels("RF", config)            # means "Random Forest"
    testCases = dataDict["testX"].copy()
    # Remove the unit because it's not a Feature
    testCases.set_index("unit", inplace=True)                # don't predict with "unit" because its not a feature
    rows = testCases.shape[0]
    predictions = np.empty(shape=[rows,1])
    for fname in models:
        model = pickle.load(open(fname, 'rb'))
        preds = model.predict(testCases)                       # get the predictions for the model
        preds = np.reshape(preds, newshape=[-1,1])
        predictions = np.append(predictions, preds, axis=1)    # adds the predictions as a column
    predictions = predictions[:,1:]                            # First column is empty (np.empty)
    return predictions