''' This is the point of entry for forecasating of revenues.
    Input file will consist of Country/Program/Customer/Driver combinations.
    Combinations are processed in isolation, i.e. no mixing of combinations occurs.
    
    This module will:
    - get the command line arguments
    - get the Config file
    - set the Logging level
    - read the input file into a dataframe. This dataframe is then passed around, \
    modified and eventually persisted
    - kick off the processing of the input file'''
    
__author__ = "The Hackett Group"

from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
import numpy as np
from setLogging import setLogging
import logging
import prepData
import forecast
import evaluate
import jobNumber
import time

def process(df, config):
    dataDict = prepData.process(df, config)
    predictions = forecast.process(dataDict)
    np.savetxt("/home/tbrownex/predictions.csv", predictions)
    return evaluate.process(predictions, dataDict)

if __name__ == "__main__":
    args     = getArgs()
    config = getConfig()
    job = jobNumber.getJob()
    
    setLogging(config)
    rec = "Start of run " + job
    logging.info(rec)
    
    start = time.time()
    df = getData(config, args)
    score = process(df, config)
    elapsed = (time.time() - start)/60
    rec = "Ended run " +job+ " after " +str(round(elapsed,1)) + " minutes with score " + str(round(score, 3))
    logging.info(rec)
    print(rec)
    
    jobNumber.setJob(int(job) + 1)