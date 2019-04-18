import pandas as pd
import numpy  as np

from getConfig   import getConfig
from getData     import getData
from preProcess  import preProcess
from optimizeRF  import buildRF
#from optimizeNN  import buildNN
from optimizeXGB import buildXGB
from optimizeSTL import buildStl
from calcRMSE      import calcRMSE

numPanels   = 3
trainMonths = 6
testMonths  = 2

def getDates(trainStart):
    ''' Compute the end of training and the Test start and end dates '''
    trainEnd   = trainStart + pd.DateOffset(months=trainMonths)
    trainEnd   = trainEnd - pd.DateOffset(days=1)
    testStart  = trainStart + pd.DateOffset(months=trainMonths)
    testEnd    = testStart + pd.DateOffset(months=testMonths)
    testEnd    = testEnd - pd.DateOffset(days=1)
    return trainEnd, testStart, testEnd

def createDatasets(data, labels, trainStart):
    ''' We're going to be processing by Date so need both features and labels indexed by date.
    Return a dictionary with trainX, trainY and testX, testY (same format as dataDict) '''
    data.set_index(["date", "hour"], inplace=True)
    idx = data.index
    labels = pd.Series(labels.values, index=idx)
    trainEnd, testStart, testEnd = getDates(trainStart)
    panelDict = {}
    panelDict["trainX"] = data.loc[trainStart   : trainEnd]
    panelDict["testX"]  = data.loc[testStart    : testEnd]
    panelDict["trainY"] = labels.loc[trainStart : trainEnd]
    panelDict["testY"]  = labels.loc[testStart  : testEnd]
    return panelDict

def getModelPreds(panelDict, config):
    ''' 
    Input is the data for a single panel
    For each entry in "optimizers" call the associated module and get its predictions
    Then compute the average of all the predictions, the "ensemble" '''
    df = pd.DataFrame()
    
    optimizers = {}
    optimizers["RF"] = buildRF
    #optimizers["NN"] = buildNN
    optimizers["XGB"] = buildXGB
    optimizers["STL"] = buildStl
    for typ, module in optimizers.items():
        preds = module(panelDict, config)
        df[typ] = preds
    df["ensemble"] = df.mean(axis=1)
    return df

def calcErrors(panel, df):
    '''
    - DataFrame has a column of predictions for each Model type
    - The last column has Actuals, which are used to compute the error (rmse and mape)
    '''
    for col in df.columns:
        if col != "actual":
            rmse = calcRMSE(df["actual"], df[col])

def processPanels(dataDict, config):
    ''' Loop through each Panel one at a time
    - "dataDict" has the entire file (all panels)
    - "panelDict" has just one panel '''
    grp = dataDict["features"].groupby(level=0)    # indexed by panel number
    trainStart = pd.to_datetime("5/25/2017")
    dfList = []
    
    for panel, data in grp:
        labels = dataDict["labels"].loc[panel]
        panelDict = createDatasets(data, labels, trainStart)
        df = getModelPreds(panelDict, config)
        df["actual"] = panelDict["testY"].values
        calcErrors(panel, df)
        df["panel"]  = panel
        idx = panelDict["testY"].index
        df.set_index(idx, inplace=True)
        dfList.append(df)
    return dfList
'''def formatResults(results):
    import operator
    import collections
    "results" is a dictionary of dictionaries: d["RF"] has keys "mape" and "rmse"
    First convert the dict of dicts to just a dict
    Then sort by rmse
    
    d = {}
    for x in results.keys():
        d[x] = results[x]["rmse"]   # ignore MAPE for now
    l = sorted(d.items(), key=operator.itemgetter(1))   # sort by rmse
    # "l" is a list of tuples (model, rmse) e.g. ("RF", 24.2)
    results = []
    for k, val in l:
        d = {}
        d[k] = val
        results.append(d)
    return results'''

if __name__ == "__main__":
    '''
    Run the optimizer routine for each panel and each model type (Random Forest, NN, XGB)
    Get the error and predictions of the best model
    Get the error for the Baseline plus the ensemble
    '''
    config   = getConfig()
    df       = getData(config)
    dataDict = preProcess(df, config)
    
    # For testing, get a few random panels
    panels = dataDict["features"].index.values
    panels = np.random.choice(panels, size=numPanels, replace=False)
    dataDict["features"] = dataDict["features"].loc[dataDict["features"].index.isin(panels)]
    dataDict["labels"]   = dataDict["labels"].loc[dataDict["labels"].index.isin(panels)]
    
    dfList = processPanels(dataDict, config)
    final = pd.concat(dfList)
    final.to_csv(config["dataLoc"]+"predictions.csv", index=True)