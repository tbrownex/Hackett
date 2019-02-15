import tensorflow as tf
import numpy  as np
import pandas as pd
import time

from pathlib import Path
from getModels import getModels
from getConfig import getConfig
from selectSet import selectSet
from getSet import getSet
from getData import getData
import prepData
from normalizeData import normalize
from evaluate import evaluate
from setLogging import setLogging
import logging

def createDataDict(Set, config):
    ''' Two separate files for Train and Test. Each file is keyed by "set" which the user has selected.
    NN requires inputs to be normalized; the other algos do not
    "train" dataset is not needed but we go through it anyway to reuse the modules
    '''
    train, test = getData(config)
    test  = getSet(test, Set)
    
    d = prepData.process(train, test, config)
    d = normalize(d, "Std")
    return d

def formatPredictions(predictions, dataDict):
    '''
    predictions is a 1-D numpy array. Need to associate the Unit and Actual in a dataframe for "evaluate"
    '''
    predictions = np.reshape(predictions, newshape=[-1,])
    actual = dataDict["testY"]
    units  = dataDict["testX"].index.values
    d = {"prediction": predictions, "actual": actual}    
    df = pd.DataFrame(data=d)
    df.set_index(units, inplace=True)
    return df
    
def getPredictions(model, dataDict):
    ''' Load the model and run Test through it '''
    tf.reset_default_graph()
    
    saver = tf.train.import_meta_graph(model)
    sess = tf.Session()
    mod = model.rstrip(".meta")
    saver.restore(sess, mod)
    
    predictions = sess.run("L3:0", feed_dict={"input:0": dataDict['testX']})
    return predictions

    # The use of the dictionary is only to match what's done in "Train"
if __name__ == "__main__":
    #args = getArgs()
    config = getConfig()
    setLogging(config)
    
    # User must select which dataset to use
    Set = selectSet()
    
    dataDict = createDataDict(Set, config)
    dataDict["testX"].set_index("unit", inplace=True)
    
    # Get the name of the saved model to use
    job = input("Enter the job ID of the saved model:")
    
    rec = "Testing NN job " + str(job) + " against set " + str(Set)
    logging.info(rec)
    
    for model in getModels("NN", job, config):
        predictions = getPredictions(model, dataDict)
        predictions = formatPredictions(predictions, dataDict)
        mape, rmse = evaluate(predictions)
        rec = str(mape) +","+ str(rmse)
        logging.info(rec)