import pandas as pd
import numpy as np
import tensorflow as tf
from getModels import getModels

def predict(model, testCases):
    ''' Load the model and run Test through it '''
    m = tf.keras.models.load_model(filepath=model, compile=False)
    predictions = m.predict(np.array(testCases))
    return predictions

def getNNpreds(dataDict, config):
    '''
    Get any NN models that have been built. Typically there will be only one model - the optimized one - but
    more are ok. For each model, generate predictions against the Test set
    
    "predictions" is a numpy array initialized empty, then you add a column for each model's predictions
    The shape of "predictions" is [rows of test cases, number of models]
    '''
    models = getModels("NN", config)            # means "Neural Network"
    testCases = dataDict["testX"].copy()
    # Remove the unit because it's not a Feature
    testCases.set_index("unit", inplace=True)
    rows = testCases.shape[0]
    predictions = np.empty(shape=[rows,1])
    for model in models:
        preds = predict(model, testCases)                       # get the predictions for the model
        preds = np.reshape(preds, newshape=[-1,1])
        predictions = np.append(predictions, preds, axis=1)    # adds the predictions as a column
    predictions = predictions[:,1:]                            # First column is empty (np.empty)
    return predictions