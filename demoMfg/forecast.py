''' Create an ensemble forecast based on several different algorithms '''

__author__ = "Tom Browne"

import numpy as np
import logging
import getRFpreds
#from getXGBparms import getXGBparms
#import getXGBpreds

def process(dataDict, config):
    '''
    Get predictions against the test set for each algo.
    The shape of "predArray" is [number of test cases (rows), number of algos]
    '''
    # Random Forest first
    predictions = getRFpreds.process(dataDict, config)
        
    # Now XGB
    return predictions