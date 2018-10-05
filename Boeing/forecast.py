''' Create an ensemble forecast based on several different algorithms

'''
__author__ = "The Hackett Group"

import numpy as np
import logging
from getRFparms import getRFparms
import getRFpreds
from getXGBparms import getXGBparms
import getXGBpreds

numModels = 4    # How many sets of predictions you'll be making

def process(dataDict):
    rows = dataDict["testX"].shape[0]
    predArray = np.empty(shape=(rows,numModels))
    
    RFparms     = getRFparms()
    XGBparms = getXGBparms()
    assert (numModels - len(RFparms) - len(XGBparms) < 0.1), "numModels must match number of parameters"
    
    idx = 0
    # Random Forest first
    for parms in RFparms:
        logging.info(parms)
        predArray[:, idx] = getRFpreds.process(parms, dataDict)
        idx += 1
        
    # Now XGB
    for parms in XGBparms:
        logging.info(parms)
        predArray[:, idx] = getXGBpreds.process(parms, dataDict)
        idx += 1
    return predArray