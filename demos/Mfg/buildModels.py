import pandas as pd
import json

from getConfig   import getConfig
from getData     import getData
from getUserCols import getCols
from preProcess  import preProcess
from optimizeRF  import buildRF
from optimizeNN  import buildNN
from optimizeXGB import buildXGB
from evaluate    import evaluate

def prepData(train, test):
    # For demo purposes user won't select the Set, just use set 1
    train = train.loc[train["set"]==1]
    test  = test.loc[test["set"]==1]
    
    svUnits = test["unit"]   # PreProcessor removes unit but we need it for seeing predictions unit-by-unit
    keep  = getCols(config)     # See what columns, if any, have been de-selected by the user
    train = train[keep]
    test  = test[keep]
    dataDict = preProcess(train, test, config)
    return dataDict, svUnits

def getModelPreds(dataDict, config):
    '''
    For each entry in "optimizers" call the associated module and get its predictions
    Also compute the average of all the predictions, the "ensemble"
    '''
    df = pd.DataFrame()
    
    optimizers = {}
    optimizers["RF"] = buildRF
    optimizers["NN"] = buildNN
    optimizers["XGB"] = buildXGB
    for typ, module in optimizers.items():
        rmse, preds = module(dataDict, config)
        df[typ] = preds
    
    df["ensemble"] = df.mean(axis=1)
    return df

def calcErrors(df):
    '''
    DataFrame has a column for each Model; the column holds the predictions
    DataFrame also has a column of Actuals, which are used to compute the error (rmse and mape)
    '''    
    df.set_index("unit", inplace=True)
    return evaluate(df)

def formatResults(results):
    import operator
    import collections
    '''
    "results" is a dictionary of dictionaries: d["RF"] has keys "mape" and "rmse"
    First convert the dict of dicts to just a dict
    Then sort by rmse
    '''
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
    return results

def writeOutput(df, results, config):
    '''
    Two files that feed the UI (C# application):
    - Predictions.csv is for viewing predictions by Unit
    - results.json is the summary statistics (error for each model type)
    '''
    df.to_csv(config["JSONloc"] + "predictions.csv")
    
    results = formatResults(results)
    with open(config["JSONloc"] + "results.json", "w") as output:
        json.dump(results, output)

if __name__ == "__main__":
    '''
    Run the optimizer routine for each model type (Random Forest, NN, XGB)
    Get the error and predictions of the best model
    Get the error for the Baseline plus the ensemble
    '''
    config            = getConfig()
    train, test       = getData(config)
    dataDict, svUnits = prepData(train, test)

    df = getModelPreds(dataDict, config)
    
    # Restore the units so we can view/track unit-by-unit error
    df["unit"]   = svUnits
    df["actual"] = dataDict["testY"]    
    results = calcErrors(df)
    writeOutput(df, results, config)