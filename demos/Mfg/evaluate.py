import numpy as np
from calcMAPE import calcMAPE
from calcRMSE import calcRMSE
from sklearn import linear_model
'''
    Calculate the MAPE and RMSE of predictions vs actuals from the Test data.
    "predictions" is a dataframe with "unit" as the index. Columns are:
        - first column is the Baseline pred and is treated different from the other, algo-based predictions
        - next few columns are the predictions from the different algos
        - last column holds the actuals
        
    "Unit" is indexed so we process the data a Unit at a time. That's how this would run in Production: you
    would be looking at a single machine's output and making a forecast for that machine, in isolation from
    the others.
    
    What prediction would you use in Production? Two choices come to mind:
    a) Take the prediction created by the last cycle
    b) Take the last X predictions and average them out
    
    "a" has a problem in that the predictions are pretty volatile from cycle to cycle.
    "b" addresses that by running a Linear Regression on the last X predictions. (I use X = 10)
    So it draws a straight line through these volatile predictions. Once you have that data regressed,
    use the last (most recent) value for your "official" prediction.
    '''
    
def getBaseline(predictions):
    errors = {}
    actuals = []
    preds   = []           
    grp = predictions.groupby(level=0)
    for unit, data in grp:
        data = data.tail(1)                # get the last row
        preds.append(data["Baseline"].iloc[0])
        actuals.append(data["actual"].iloc[0])
    actuals = np.array(actuals)
    preds   = np.array(preds)
    errors["mape"] = calcMAPE(actuals, preds)
    errors["rmse"] = calcRMSE(actuals, preds)
    return errors

def getEnsemble(predictions, typ):
    errors = {}
    actuals = []
    preds   = []
    
    X = np.array([x for x in range(10)])
    X = np.reshape(X, [-1,1])
    
    regr = linear_model.LinearRegression()
    
    grp = predictions.groupby(level=0)
    # "data" will contain a column for each prediction plus the Actual
    for unit, data in grp:
        data = data.tail(10)                # get the last 10 rows for the Regression data
        actuals.append(data["actual"].iloc[-1])
        del data["actual"]                    # Don't include the actual in the ensemble of predictions
        del data["Baseline"]                  # Don't want the Baseline included either
        ensemble = data.mean(axis=1)          # axis=1 gets the mean across rows of different algos
        if typ == "LR":
            ensemble = ensemble.values.reshape([-1,1])
            regr.fit(X, ensemble)
            LR = regr.predict(X)
            preds.append(LR[-1][0])
        else:
            preds.append(ensemble.iloc[-1])
    actuals = np.array(actuals)
    preds   = np.array(preds)
    errors["mape"] = calcMAPE(actuals, preds)
    errors["rmse"] = calcRMSE(actuals, preds)
    return errors

def evaluate(predictions, typ):
    errors = {}
    errors["baseline"] = getBaseline(predictions)
    errors["ensemble"] = getEnsemble(predictions, typ)
    return errors