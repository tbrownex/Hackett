import pandas as pd
import numpy  as np
import os
import sys
import time

from getConfig import getConfig
from getArgs import getArgs
from getData import getData
from getModelParms import getParms
import prepData
from nn import run
import jobNumber as job
from selectSet import selectSet
from getSet import getSet
from normalizeData import normalize

def createVal(d):
    # Split Training into Train and Val. They are already shuffled so just take bottom 20% for Val
    valSize = int(d["trainX"].shape[0]*.2)
    trainSize = d["trainX"].shape[0] - valSize
    d["trainX"] = d["trainX"].head(trainSize)
    d["trainY"] = d["trainY"].head(trainSize)
    d["valX"] = d["trainX"].tail(valSize)
    d["valY"] = d["trainY"].tail(valSize)
    
    del d['testX']["unit"]
    return d

def createDataDict(Set, config):
    ''' Two separate files for Train and Test. Each file is keyed by "set" which the user has selected.
    NN requires inputs to be normalized; the other algos do not
    NN requires a validation set; the other algos do not
    '''
    train, test = getData(config)
    train = getSet(train, Set)
    test  = getSet(test, Set)
    
    d = prepData.process(train, test, config)
    d = normalize(d, "Std")
    d = createVal(d)
    return d

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results, job_id):
    with open("/home/tbrownex/summary_"+str(job_id)+".txt", 'w') as summary:
        keys = results[0][1]
        hdr = "Run" +"|" + "|".join(keys)
        hdr += "|"+"RMSE" + "\n"
        summary.write(hdr)        
        
        for x in results:
            rec = str(x[0]) +"|"
            rec += "|".join([str(t) for t in x[1].values()])
            rec += "|"+ str(x[2]) +"\n"         # lift
            summary.write(rec)
            
if __name__ == "__main__":
    #args = getArgs()
    config = getConfig()
    jobId = job.getJob()
    
    # User must select which dataset to use
    Set = selectSet()
    
    dataDict = createDataDict(Set, config)
    
    parms = getParms("NN")       # The hyper-parameter combinations to be tested
    
    results = []
    count = 1
    
    start_time = time.time()
    
    for x in parms:
        parmDict = {}                  # holds the hyperparameter combination for one run
        parmDict['l1_size']       = x[0]
        #parmDict['l2_size']       = x[1]
        parmDict['learning_rate'] = x[1]
        parmDict['lambda']        = x[2]
        parmDict['batch_size']    = x[3]
        parmDict['epochs']        = x[4]
        parmDict['activation']    = x[5]
        parmDict['stdDev']        = x[6]
        
        jobName = jobId + "_" + str(count)
        
        rmse = run(dataDict, parmDict, jobName, config)
        
        tup = (count, parmDict, rmse)
        results.append(tup)
        count +=1
            
    # Write out a summary of the results
    writeResults(results, jobId)
    jobId = int(jobId)
    job.setJob(jobId+1)
    print("Job {} complete after {:,.0f} minutes".format(str(jobId), (time.time() -start_time)/60))