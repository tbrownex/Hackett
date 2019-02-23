import pandas as pd
import numpy  as np
import os
import sys
import time

from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
from getModelParms import getParms
from preProcess import preProcess
from nn         import run
from calcMAPE   import calcMAPE
from calcRMSE   import calcRMSE
import jobNumber as job

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    delim = ","
    with open("/home/tbrownex/NNscores.csv", 'w') as summary:
        hdr = "L1"+delim+"Lambda"+delim+"activation"+delim+"batchSize"+delim+"LR"+\
        delim+"StdDev"+delim+"MAPE"+delim+"RMSE"+"\n"
        summary.write(hdr)
        
        for x in results:
            rec = str(x[0][0])+delim+str(x[0][1])+delim+str(x[0][2])+\
            delim+str(x[0][3])+delim+str(x[0][4])+delim+str(x[0][5])+\
            delim+str(x[1])+delim+str(x[2])+"\n"
            summary.write(rec)

if __name__ == "__main__":
    args     = getArgs()
    config   = getConfig()
    df       = getData(config)
    dataDict = preProcess(df, config, args)
    
    jobId = job.getJob()
    
    parms = getParms("NN")       # The hyper-parameter combinations to be tested
    
    results = []
    count = 1
    
    start_time = time.time()
    print("\n{} parameter combinations".format(len(parms)))
    print("\n{:<10}{:<10}{}".format("Count", "MAPE","RMSE"))
    
    for x in parms:
        parmDict = {}                  # holds the hyperparameter combination for one run
        parmDict['l1Size']      = x[0]
        parmDict['lambda']      = x[1]
        parmDict['activation']  = x[2]
        parmDict['batchSize']   = x[3]
        parmDict['lr']          = x[4]
        parmDict['std']         = x[5]
        
        jobName = "job_" + jobId +"/"+ "run_" + str(count)
        
        preds = run(dataDict, parmDict, config, jobName)
        
        mape = calcMAPE(dataDict["testY"], preds)
        rmse = calcRMSE(dataDict["testY"], preds)
        
        print("{:<10}{:<10.1%}{:.2f}".format(count, mape, rmse))
        tup = (x, mape, rmse)
        results.append(tup)
        count +=1
        
    # Write out a summary of the results
    writeResults(results)
    
    jobId = int(jobId)
    job.setJob(jobId+1)
    print("\nJob {} complete after {:,.0f} minutes".format(str(jobId), (time.time() -start_time)/60))