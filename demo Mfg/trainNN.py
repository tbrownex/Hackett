import pandas as pd
import numpy  as np
import os
import sys
import time

from getConfig import getConfig
from getArgs import getArgs
from getData import getData
from prepDataNN import prepData
from nn import run
import jobNumber      as job

# NN hyper-parameters
l1_size       = [256]          # Count of nodes in layer 1
learning_rate = [0.01]
Lambda        = [0]          # Regularization parameter
batch_size    = [64]
epochs        = [20]
activation    = ['tanh']           # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'
stdDev    = [0.01]            # StdDev for initializing weights

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
    args = getArgs()
    config = getConfig()
    df = getData(config)
    dataDict = prepData(df, config, args)
    jobId = job.getJob()
    
    for x in activation:
        assert x in ['tanh', 'leakyReLU', 'ReLU', 'ReLU6'], "Invalid Activation: %s" % x
        
    # "parms" holds all the combinations of hyperparameters
    parms = [[a,b,c,d,e,f,g] for a in l1_size
             for b in learning_rate
             for c in Lambda
             for d in batch_size
             for e in epochs
             for f in activation
             for g in stdDev]
    
    results = []                    # holds the hyperparameters and results for each run
    start_time = time.time()
    
    count = 1
    for x in parms:
        parmDict = {}                  # holds the hyperparameter combination for one run
        parmDict['l1_size']       = x[0]
        parmDict['learning_rate'] = x[1]
        parmDict['lambda']        = x[2]
        parmDict['batch_size']    = x[3]
        parmDict['epochs']        = x[4]
        parmDict['activation']    = x[5]
        parmDict['stdDev']    = x[6] 
            
        job_name = "job_" + jobId +"/"+ "run_" + str(count)
        
        rmse = run(dataDict, parmDict, job_name, config)
        
        tup = (count, parmDict, rmse)
        results.append(tup)
        count +=1
            
    # Write out a summary of the results
    writeResults(results, jobId)
    jobId = int(jobId)
    job.setJob(jobId+1)
    print("Job {} complete after {:,.0f} minutes".format(str(jobId), (time.time() -start_time)/60))