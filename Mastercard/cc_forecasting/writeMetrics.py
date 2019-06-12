# -*- coding: utf-8 -*-
"""
This is a dummy function
its purpose is to give a signal to the listener that
all other tasks have been completed and it is okay to
write the file to disk

The listener uses the inspect library to check who the caller is and
if the caller is writeMetrics then it will trigger a call to abstractWriter 

Created on Wed Sep 26 17:03:18 2018

@author: William Cairns
"""
import numpy as np
import pandas as pd
from metricsCollector import listener

def writeMetrics(df, config):
    data = np.zeros([3,3], dtype=int).T
    dfOutput = pd.DataFrame(data)

    listener(dfOutput)
    
    return(df)
    
