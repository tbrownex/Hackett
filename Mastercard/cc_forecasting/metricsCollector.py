#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:13:14 2018
The metrics collector is a module containing
a single function called listener. The purpose
of this function is to iteratively receive dataframes
from each of the functions of the abstract descriptor class
such as five-number, trend, anomaly, change point, stationarity.
As each descriptor completes its work it calls the
metrics collector which appends the new data to the variable
called metricsDf. metricsDf is initialized globally using
settings.py from main.py. On each iteration the metricsDf is appended
until the calling function is writeMetrics which is the signal to the listener
that all metrics are complete and the results can be passed to the abstract
writer to be written to disk using the proper filename formatting. 

@author: wjc
"""
import inspect
import pandas as pd
import numpy as np
import settings
from abstractWriter import csvWriter
from getConfig import getConfig

config = getConfig()

def listener(df):
     
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    
    if calframe[1][3] == 'writeMetrics':
        #return 'Main-1'
        print('Metrics Data Complete')
        dfOutput = settings.dfMetrics
        csvWriter(dfOutput,'descriptStats',config)
    else:
        settings.dfMetrics = pd.concat([settings.dfMetrics,df])
        print('Metrics Data Received')
        
    #dfOutput = settings.dfMetrics
    
    #print(dfOutput)
    
    return
    
    
