''' This is the point of entry for forecasating.
    Assumption is that the individual algorithms have had some parameter optimization done already
    and models have been generated and stored in some folder
    Two input files: one holds Training data and the other Test
    Data format:
       - "set" splits the data into 4 groups; you can't mix the groups
       - "unit" groups a bunch of readings for a specific machine
       - "cycle" is a single operational cycle; could be something like one-hour continuous operation
       - "settings" and "sensors" are the diagnostics which will serve as features
       - "RUL" is measured in cycles and serves as the label
       
    This module will:
    - get the command line arguments
    - get the Config file
    - set the Logging level
    - read the input files
    - prep the data and put into a dictionary
    - kick off the processing '''
    
__author__ = "Tom Browne"

#from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
import pandas as pd
import numpy as np
from setLogging import setLogging
from selectSet import selectSet
from getSet import getSet
import logging
import prepData
import forecast
import evaluate
import jobNumber
import time

def process(train, test, config):
    '''
    - Prep the data (split features from labels)
    - Get the predictions from each model
    - Evaluate the predictions (calculate error)
    
    dataDict has keys for trainX, trainY and testX, testY
    '''
    dataDict = prepData.process(train, test, config)
    predictions = forecast.process(dataDict, config)
    '''
    "evaluate" module expects a dataframe with:
    - Unit
    - A column for each algo's predictions
    - Actual RUL
    '''
    predDF = pd.DataFrame(predictions)
    predDF["unit"]   = dataDict["testX"]["unit"]
    predDF["actual"] = dataDict["testY"]
    predDF.to_csv("/home/tbrownex/predDF.csv", index=False)
    predDF.set_index("unit", inplace=True)
    return evaluate.process(predDF)

def finishUp(start, mape, rmse, job):
    '''
    - Print the run time and error metrics
    - Update the log
    - Update the Job number
    '''
    elapsed = (time.time() - start)/60
    rec = "Ended run " +job+ " after " +str(round(elapsed,1)) + " minutes with MAPE " +\
    str(round(mape, 3)) + " and RMSE " + str(round(rmse,2))
    logging.info(rec)
    print(rec)
    
    jobNumber.setJob(int(job) + 1)

if __name__ == "__main__":
    '''
    - Get the Job number for this job
    - Set up logging
    - User will specify the dataset to use
    - Load Train and Test dataframes with user-specified dataset
    - Kick off processing
    '''
    #args     = getArgs()
    config = getConfig()
    # Do some initialization
    job = jobNumber.getJob()
    setLogging(config)
    rec = "Start of run " + job
    logging.info(rec)
    
    start = time.time()
    # Which dataset should we work with?
    Set = selectSet()

    train, test = getData(config)
    train = getSet(train, Set)
    test = getSet(test, Set)
    
    mape, rmse = process(train, test, config)
    
    finishUp(start, mape, rmse, job)