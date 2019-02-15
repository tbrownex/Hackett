import pandas as pd
import numpy  as np
import time
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from getConfig import getConfig
from getData import getData
from getModelParms import getParms
from selectSet import selectSet
from getSet import getSet
from preProcess import preProcess
from getArgs import getArgs
from evaluate import evaluate

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

def formatPreds(dataDict, svUnits, preds):
    ''' Prepare the data to be evaluated '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["pred"]   = preds
    d["unit"]   = svUnits
    df = pd.DataFrame(d)
    df.set_index("unit", inplace=True)
    return df

def process(dataDict, parms, config):
    # Remove Unit since its not a feature but save it so you can recreate
    # the Predictions vs Actuals by Unit
    svUnits = dataDict["testX"]["unit"]
    del dataDict["testX"]["unit"]
    
    results = []
    count = 1
    
    for x in parms:
        params = {'n_estimators': x[0],\
                  'min_samples_split': x[1],\
                  'max_depth': x[2],\
                  "min_samples_leaf": x[3],\
                  "max_features": x[4]}
        regr  = RandomForestRegressor(**params)
        regr.fit(dataDict["trainX"], dataDict["trainY"])
        preds = regr.predict(dataDict["testX"])
        df    = formatPreds(dataDict, svUnits, preds)
        mape, rmse = evaluate(df, config["evaluationMethod"])
        print("{}{:<8.2f}{}{:.2f}".format("mape ", mape, "rmse: ", rmse))
        tup = (x, mape, rmse)
        results.append(tup)
        saveModel(regr, count)
        print("Done with {} of {}".format(count, len(parms)))
        count += 1
    return results
    
if __name__ == "__main__":
    args   = getArgs()
    config = getConfig()
    
    Set = selectSet()

    train, test = getData(config)
    train       = getSet(train, Set)
    test        = getSet(test, Set)
    
    dataDict    = preProcess(train, test, config, args)
    
    parms = getParms("RF")
    
    start_time = time.time()
    
    results = process(dataDict, parms, config)
            
    # Write out a summary of the results
    writeResults(results)
    print("Done after {:,.0f} minutes".format((time.time() -start_time)/60))