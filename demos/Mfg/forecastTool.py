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

import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore')

from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
from setLogging import setLogging
from selectSet import selectSet
from getSet import getSet
import logging
from preProcess import preProcess
import forecast
import jobNumber
from evaluate import evaluate

def process(dataDict, config):
    '''
    - Get the predictions from each model
    - Evaluate the predictions (calculate error)
    '''
    predDF = forecast.process(dataDict, config)
    # Add the unit and actuals
    predDF["unit"]   = dataDict["testX"]["unit"]
    predDF["actual"] = dataDict["testY"]
    predDF.to_csv("/home/tbrownex/predDF.csv", index=False)
    predDF.set_index("unit", inplace=True)
    return evaluate(predDF, config["evaluationMethod"])

def finishUp(errors, job):
    '''
    - Print the run time and error metrics
    - Update the log
    - Update the Job number
    '''
    for k in errors.keys():
        print(k, ": ", errors[k])

if __name__ == "__main__":
    '''
    - Get the Job number for this job
    - Set up logging
    - User will specify the dataset to use
    - Load Train and Test dataframes with user-specified dataset
    - Kick off processing
    '''
    args   = getArgs()
    config = getConfig()
    # Do some initialization
    job = jobNumber.getJob()
    setLogging(config)
    rec = "Start of run " + job
    logging.info(rec)
    
    # Which dataset should we work with?
    Set = selectSet()

    train, test = getData(config)
    train = getSet(train, Set)
    test = getSet(test, Set)
    
    dataDict = preProcess(train, test, config, args)
    errors   = process(dataDict, config)
    
    print("\nLearing Rate: ", config["evaluationMethod"])
    finishUp(errors, job)