# -*- coding: utf-8 -*-
"""
For each UUID compute a trend metric and write to output
"""

__author__ = "The Hackett Group"

import pandas as pd
import numpy as np
from getConfig  import getConfig
from timestampGenerator import tstamp



def csvWriter(df,functionCaller,config):
    
    """
    -Writer waits for a complete dataframe that should be saved to disk
    -Writer gets its output location from config
    -Writer appends the timestamp function in the output filename
    -Writer saves file
    """
    #print(config['outputDir'])
    outputFilename = config['outputDir'] + functionCaller + '_' + tstamp() + '.csv'
    #print(outputFilename)
    
    df.to_csv(outputFilename,index=False)

    return