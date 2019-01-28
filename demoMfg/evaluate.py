import numpy as np
from calcMAPE import calcMAPE
from calcRMSE import calcRMSE

def evaluate(predictions):
    '''
    Calculate the MAPE and RMSE of predictions vs actuals from the Test data.
    "predictions" is a dataframe with "unit" as the index. Columns are:
        - first few columns are the predictions from the different algos
        - last column holds the actuals
        
    There are different ways of computing the error. One way would be to ignore unit and calculate
    the error for every cycle for every unit. That's the easiest but the prediction when there are a
    large number of cycles to go is not as valuable or important as when the prediction is lower i.e.
    sooner to fail. So mainly I get the error towards the end of the cycles (which is a per-unit error).
    '''
    '''
    This is to get the error for all cycles, all units
    actuals = predictions["actual"]
    preds   = predictions[0]
    mape = calcMAPE(actuals, preds)
    rmse = calcRMSE(actuals, preds)
    '''
    # Get the error for last cycle in each unit
    actuals = []
    preds   = []
    grp = predictions.groupby(level=0)
    for unit, data in grp:
        last = data.tail(1)                # tail(1) means get the last row (last cycle)
        actuals.append(last["actual"].iloc[0])
        del last["actual"]                    # Don't include the actual in the ensemble of predictions
        ensemble = last.mean(axis=1)          # axis=1 gets the mean across rows of different algos
        preds.append(ensemble.iloc[0])
    actuals = np.array(actuals)
    preds   = np.array(preds)
    mape = calcMAPE(actuals, preds)
    rmse = calcRMSE(actuals, preds)
    return mape, rmse