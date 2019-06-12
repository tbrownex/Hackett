# -*- coding: utf-8 -*-
"""
This is the point of entry for forecasting of revenues.
Input file will consist of Country/Program/Customer/Driver combinations.
    
This module will:
    - get the command line arguments
    - get the Config file
    - set the Logging level
    - read the input file into a dataframe. This dataframe is then dispatched
    modified processed and eventually persisted
"""

__author__ = 'The Hackett Group'

# Python Imports
import numpy as np
import pandas as pd
import os
import sys
import csv
import time

# These are local environment specific and will need modification
# for each system
sys.path.insert(0,'C:/Users/e084332/Documents/repos/sandbox')
sys.path.insert(1,'C:/Users/e084332/Documents/repos/cc_forecasting')
sys.path.insert(2,'/Users/wjc/Documents/repos/sandbox')
sys.path.insert(3,'/Users/wjc/Documents/repos/cc_forecasting')


# Project Imports
import settings
from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
from setLogging import setLogging
from timestampGenerator import tstamp
from descriptStats import descript
from stationarity import dickey_fuller
from filterData import filterData
from normalize import z_score
from anomalyCount import anomalyModel
from analyzeSlope import slope
from writeMetrics import writeMetrics
from metricsCollector import listener
from arimaModel import trainArima
from arimaModel import predictArima
from bstsModel import trainBSTS
from bstsModel import predictBSTS
from etsModel import trainETS
from etsModel import predictETS
from abstractWriter import csvWriter
from abstractCollaborator import bestModel
import logging
import preprocessor

settings.init()
start_time = time.time()

if __name__ == "__main__":    
    args   = getArgs()
    config = getConfig()
    setLogging(config, args)
    logging.info("Start of a run")
    
    
    # Read the import datafile using parameters from the config
    df = getData(config)


    #for x in range(1, 4):
    #    dfTemp = pd.DataFrame(np.zeros([3,3],dtype=int).T)
    #    dfTemp = dfTemp.replace(0,x)
    #    csvWriter(dfTemp,'testing',config)
        

    # Abstract Dispatcher Class (simplified)
    #descript will not need to return the df because it's
    #function is to gather necessary stats then send them to
    #output. Filter will need to return the data because it will
    # have changed.
    #task_list = [descript, dickey_fuller, anomalyModel, slope, writeMetrics, filterData, z_score]
    task_list = [trainArima, trainETS, trainBSTS, bestModel, predictArima, predictETS, predictBSTS]
    
    for task in task_list:
        df = task(df,config)
        print('-------' + str(task) + '--------')
        print(df.shape)
        print("--- %s seconds ---" % (time.time() - start_time))

    
    logging.info("End of a run")
    
    #with open('C:/Users/e084332/Documents/RevenueForecasting/dict.csv', 'w') as csv_file:
    #    writer = csv.writer(csv_file)
    #    for key, value in settings.collabDict.items():
    #       writer.writerow([key, value])


