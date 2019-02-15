from numpy.random import seed
seed(1234)

import pandas as pd
import numpy  as np
import time
from sklearn.ensemble import RandomForestRegressor
import pickle

from getArgs       import getArgs
from getConfig     import getConfig
from getData       import getData
from getModelParms import getParms
from preProcess    import preProcess
from calcMAPE  import calcMAPE
from calcRMSE  import calcRMSE
from evalRFfeatures import evalFeatures

def saveModel(model, count):
    fname = "RFmodel_" + str(count)
    pickle.dump(model, open(config["modelDir"] + fname, 'wb'))

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    delim = ","
    with open("/home/tbrownex/RFscores.csv", 'w') as summary:
        hdr = "trees"+delim+"nodeSize"+delim+"depth"+delim+"leafSize"+delim+"features"+\
        delim+"MAPE"+delim+"RMSE"+"\n"
        summary.write(hdr)
        
        for x in results:
            rec = str(x[0][0])+delim+str(x[0][1])+delim+str(x[0][2])+\
            delim+str(x[0][3])+delim+str(x[0][4])+delim+str(x[1])+delim+str(x[2])+"\n"
            summary.write(rec)

def process(dataDict, parms, config):    
    results = []
    count = 1
    
    print("{:<8}{:<10}{}".format("Count", "MAPE", "RMSE"))
    
    for x in parms:
        params = {'n_estimators': x[0],\
                  'min_samples_split': x[1],\
                  'max_depth': x[2],\
                  "min_samples_leaf": x[3],\
                  "max_features": x[4]}
        regr  = RandomForestRegressor(**params)
        regr.fit(dataDict["trainX"], dataDict["trainY"])
        preds = regr.predict(dataDict["testX"])
        
        mape = calcMAPE(dataDict["testY"], preds)
        rmse = calcRMSE(dataDict["testY"], preds) 
        print("{:<8}{:<10.2f}{:.2f}".format(count, mape, rmse))
        tup = (x, mape, rmse)
        results.append(tup)
        #saveModel(regr, count)
        if count % 10 == 0:
            print("Done with {} of {}".format(count, len(parms)))
        count += 1
    return results
    
if __name__ == "__main__":
    args   = getArgs()
    config = getConfig()

    df       = getData(config)
    dataDict = preProcess(df, config, args)
    
    # Show the feature ranking
    if args.Features == "Y":
        evalFeatures(dataDict)
        input()
    
    
    parms = getParms("RF")
    start_time = time.time()
    
    results = process(dataDict, parms, config)
            
    # Write out a summary of the results
    writeResults(results)
    print("Done after {:,.0f} minutes".format((time.time() -start_time)/60))