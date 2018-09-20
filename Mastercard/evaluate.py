''' Calculate the MAPE of predictions vs test.

"predictions" is a numpy array
"test" is a dataframe'''

import pandas as pd
import numpy as np

def process(predictions, test):
    test = np.array(test["Amount"])
    
    '''This is a workaround a problem with STL: the forecast periods returned do not match
    what I have specified. So need to truncate the forecast period'''
    predictions = predictions[:test.shape[0]]
    
    return np.mean(np.abs((predictions - test) / test))