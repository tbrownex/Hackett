''' Create an ensemble forecast based on several different algorithms '''

__author__ = "Tom Browne"

import numpy as np
import logging
from getRFpreds import getRFpreds
from getNNpreds import getNNpreds
#from getXGBparms import getXGBparms
#import getXGBpreds

def process(dataDict, config):
    '''
    Get predictions against the test set for each algo.
    The shape of "predictions" that is returned is [number of test cases (rows), number of algos]
    '''
    # Random Forest first
    predictions = getRFpreds(dataDict, config)
    
    # Neural Network
    NNpreds     = getNNpreds(dataDict, config)
    predictions = np.append(predictions, NNpreds, axis=1)
        
    # Now XGB
    return predictions