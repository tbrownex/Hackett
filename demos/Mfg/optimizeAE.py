from numpy.random import seed
seed(1814)
from tensorflow import set_random_seed
set_random_seed(1814)

import pandas as pd
import numpy  as np
import os
import sys
import time
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getConfig import getConfig
from getData import getData
from getModelParms import getParms
from preProcess import preProcess
from kerasAE import runNN
from selectSet import selectSet
from getSet import getSet

def loadParms(p):
    params = {'l1Size':     p[0],
              'l2Size':     p[1],
              'activation': p[2],
              'batchSize':  p[3],
              'lr':         p[4],
              'std':        p[5],
              'dropout':    p[6],
              'optimizer':  p[7]}
    return params

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    delim = ","
    keys = results[0][0].keys()
    hdr = delim.join(keys)
    hdr += delim+"MAPE"+delim+"MSE"+"\n"
    
    with open("/home/tbrownex/NNscores.csv", 'w') as summary:
        summary.write(hdr)
        
        for x in results:
            vals = list(x[0].values())
            rec = delim.join(str(v) for v in vals)
            mape = str(round(x[1],2))
            mse  = str(round(x[2],2))
            rec += delim+ mape +delim+ mse +"\n"
            summary.write(rec)

def formatPreds(dataDict, svUnits, preds):
    ''' Prepare the data to be evaluated '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["pred"]   = np.reshape(preds, [-1,])
    d["unit"]   = svUnits
    df = pd.DataFrame(d)
    df.set_index("unit", inplace=True)
    return df

if __name__ == "__main__":
    config = getConfig()
    Set = selectSet()

    train, test = getData(config)
    train       = getSet(train, Set)
    test        = getSet(test, Set)
    
    dataDict    = preProcess(train, test, config)
    
    # Add back the label which was split out: for autoencoder and outlier detection, you should
    # include the Label as part of the data
    dataDict["trainX"]["RUL"] = dataDict["trainY"]
    dataDict["testX"]["RUL"]  = dataDict["testY"]
    
    parms = getParms("AE")       # The hyper-parameter combinations to be tested
    
    results = []
    count = 1
    
    start_time = time.time()
    print("\n{} parameter combinations".format(len(parms)))
    print("{:<6}{:<10}{}".format("Count", "MAPE","MSE"))
    
    for p in parms:
        parmDict = loadParms(p)
        mape, mse = runNN(dataDict, parmDict, config, count)
        
        print("{:<6}{:<10.2f}{:.2f}".format(count, mape, mse))
        tup = (parmDict, mape, mse)
        results.append(tup)
        count +=1
            
    # Write out a summary of the results
    writeResults(results)
    print("\nComplete after {:,.0f} minutes".format((time.time() -start_time)/60))