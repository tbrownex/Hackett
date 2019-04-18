import pandas as pd
import numpy  as np
import os
import sys
import time

from getConfig  import getConfig
#from getData    import getData
from preProcess import preProcess
from getModelParms import getParms
from nn            import run

def getData(config, rows=None):
    ''' Specify "cols" and "dtypes" to minimize memory usage, otherwise memory blows up '''
    cols = ["age_alt","income_alt","marital_status","owner_type",\
            "home_value","num_hh","num_adults","num_kids","net_worth_alt","density",\
            "age_bins","income_bins","hv_bins","density_bins","white","black",\
            "hispanic","asian","jewish","indian","other","neighborhood_bin", "Label"]
    sep = "|"
    types={'Label':"int16",
          'age_alt': "int16",\
          'income_alt': "int16",\
          'marital_status': "int16",\
          'owner_type': "int16",\
          'home_value': "int16",\
          'num_hh': "int16",\
          'num_adults': "int16",\
          'num_kids': "int16",\
          'net_worth_alt': "float64",\
          'density': "float64",\
          'age_bins': "int16",\
          'income_bins': "int16",\
          'hv_bins': "int16",\
          'density_bins': "int16",\
          'white': "int16",\
          'black': "int16",\
          'hispanic': "int16",\
          'asian': "int16",\
          'jewish': "int16",\
          'indian': "int16",\
          'other': "int16",\
          'neighborhood_bin': "int16"}
    return pd.read_csv(config["dataLoc"]+config["fileName"],\
                       sep=sep,\
                       nrows=rows,\
                       usecols=cols,\
                       dtype=types)

def loadParms(p):
    params = {'l1Size':     p[0],
              'activation': p[1],
              "Lambda":     p[2],
              'batchSize':  p[3],
              'lr':         p[4],
              'std':        p[5],
              'dropout':    p[6],
              'optimizer':  p[7],
              "weight":     p[8]}
    return params

def print_sales_ratio(dataDict):
    print("\n{:<55}{:<12}{}".format("Prevalence of Positives in the different datasets","Dataset", "ratio"))
    for set in ["trainY", "valY", "testY"]:
        pos = np.sum(dataDict[set][:,1])
        tot = dataDict[set].shape[0]
        print("{:<55}{:<12}{}{}".format("",set, int(tot/pos), ":1"))

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    with open("/home/tbrownex/summary.txt", 'w') as summary:
        parm = results[0][0]
        hdr = "|".join(parm.keys())
        hdr += "|" +"Score" + "\n"
        summary.write(hdr)        
        
        for x in results:
            parms = x[0].values()
            score = round(x[1],4)
            rec = "|".join([str(t) for t in parms])
            rec += "|"+ str(score) +"\n"
            summary.write(rec)

def process(dataDict, parms, config):
    bestRMSE = np.inf
    bestPreds = None
    results = []
    for p in parms:
        parmDict = loadParms(p)
        mse = run(dataDict, parmDict, config)
        tup = (parmDict, mse)
        results.append(tup)
        '''if rmse < bestRMSE:
            bestRMSE  = rmse
            bestPreds = preds
            saveModel(nn, config)'''
    return results
    #return bestRMSE, bestPreds

if __name__ == "__main__":
    config = getConfig()
    df = getData(config, rows=100000)    # Can't use the "getData" module in /common because of memory limitation
    dataDict = preProcess(df, config)
    print_sales_ratio(dataDict)
    
    parms = getParms("NN")
    
    results = process(dataDict, parms, config)
    writeResults(results)
    #rmse, preds = process(dataDict, parms, config)