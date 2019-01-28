import numpy as np
import pandas as pd
import itertools
from getConfig  import getConfig
import prepData
import getXGBpreds
import evaluate
import time

def getData(config):
    df = pd.read_csv(config["dataLoc"]+config["inputFile"])
    
    df = df.sample(frac=0.1)
    return prepData.process(df, config)

def getConfigs():
    trees    =  [100]
    depth    = [6, 8]
    lr       = [x for x in np.linspace(start=0.09, stop=0.3, num=5)]
    sample   = [.25, 0.5,]
    features = [0.7, 1.0]
    state    = [1919]
    
    return itertools.product(trees, depth, lr, sample, features, state)

def process(dataDict, parmList):
    dfList = []
    
    for p in parmList:
        trees, depth, lr, sample, features, state = p
        parms = {
            "n_estimators":     trees,
            "max_depth":        depth,
            "learning_rate":    lr,
            "subsample":        sample,
            "colsample_bytree": features,
            "random_state": state}    
        predictions = getXGBpreds.process(parms, dataDict)
    
        parms["score"] = np.mean(np.abs((predictions - dataDict["testY"]) / dataDict["testY"]))
        tmp = pd.DataFrame.from_records([parms])
        dfList.append(tmp)

    return  pd.concat(dfList)

if __name__ == "__main__":
    config = getConfig()
    dataDict = getData(config)
    parmList = getConfigs()
    
    start = time.time()
    results = process(dataDict, parmList)
    results.to_csv("/home/tbrownex/XGBresults.csv", index=False)
    elapsed = (time.time() - start)/60
    print("Elapsed time: {:.1f} minutes".format(elapsed))