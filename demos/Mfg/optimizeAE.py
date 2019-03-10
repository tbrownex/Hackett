import pandas as pd
import numpy  as np
import os
import sys
import time
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getConfig import getConfig
from getArgs import getArgs
from getData import getData
from getModelParms import getParms
from preProcess import preProcess
from kerasAE import runNN
import jobNumber as job
from selectSet import selectSet
from getSet import getSet
from normalizeData import normalize

def createVal(d):
    # Split Training into Train and Val. They are already shuffled so just take bottom 20% for Val
    valSize = int(d["trainX"].shape[0]*.2)
    trainSize = d["trainX"].shape[0] - valSize
    d["trainX"] = d["trainX"].head(trainSize)
    d["valX"] = d["trainX"].tail(valSize)
    
    return d

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    delim = ","
    with open("/home/tbrownex/NNscores.csv", 'w') as summary:
        hdr = "L1"+delim+"activation"+delim+"batchSize"+delim+"LR"+\
        delim+"StdDev"+delim+"Dropout"+delim+"optimizer"+delim+"MAPE"+delim+"RMSE"+"\n"
        summary.write(hdr)
        
        for x in results:
            rec = str(x[0][0])+delim+str(x[0][1])+delim+str(x[0][2])+\
            delim+str(x[0][3])+delim+str(x[0][4])+delim+str(x[0][5])+\
            delim+str(x[0][6])+delim+str(x[1])+delim+str(x[2])+"\n"
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
    args   = getArgs()
    config = getConfig()
    Set = selectSet()

    train, test = getData(config)
    train       = getSet(train, Set)
    test        = getSet(test, Set)
    
    dataDict    = preProcess(train, test, config, args)
    dataDict    = createVal(dataDict)
    
    # Remove Unit since its not a feature
    del dataDict["testX"]["unit"]
    
    parms = getParms("AE")       # The hyper-parameter combinations to be tested
    
    results = []
    count = 1
    
    start_time = time.time()
    print("\n{} parameter combinations".format(len(parms)))
    print("{:<6}{:<10}{}".format("Count", "MAPE","RMSE"))
    
    for x in parms:
        parmDict = {}                  # holds the hyperparameter combination for one run
        parmDict['l1Size']      = x[0]
        parmDict['l2Size']      = x[1]
        parmDict['activation']  = x[2]
        parmDict['batchSize']   = x[3]
        parmDict['lr']          = x[4]
        parmDict['std']         = x[5]
        parmDict['dropout']     = x[6]
        parmDict['optimizer']   = x[7]
        
        mape, rmse = runNN(dataDict, parmDict, config)
        
        print("{:<6}{:<8.2f}{:.2f}".format(count, mape, rmse))
        tup = (x, mape, rmse)
        results.append(tup)
        count +=1
            
    # Write out a summary of the results
    writeResults(results)
    print("Complete after {:,.0f} minutes".format((time.time() -start_time)/60))