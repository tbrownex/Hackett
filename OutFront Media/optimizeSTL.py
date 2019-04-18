import pandas as pd
import numpy  as np
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

from getModelParms import getParms
from calcRMSE      import calcRMSE

''' def saveModel(model, config):
    pickle.dump(model, open(config["modelDir"] + "RFmodel", 'wb'))'''

'''def decompose(model, length):
    matrix = model.rx2('time.series')
    decomp = [x for x in matrix]
    
    df              = pd.DataFrame()
    df['trend']     = decomp[length:2*length]
    df['seasonal']  = decomp[0:length]
    df['residuals'] = decomp[2*length:3*length]
    return df'''

def loadParms(p):
    d = {}
    d["window"] = p[0]
    return d

def process(panelDict, parms, config):
    values = list(panelDict["trainY"])
    length = len(values)
    values = r.ts(values, frequency=24)
    
    bestRMSE = np.inf
    bestPreds = None
    for p in parms:
        d = loadParms(p)
        model  = r.stl(values, s_window = d["window"])
        
        '''df = decompose(model, length)
        df["actual"] = panelDict["trainY"].values
        df.to_csv("/home/tbrownex/"+"STLmodel.csv", index=False)'''
        
        forecast      = importr('forecast')
        forecastCount = len(panelDict["testX"])
        fcast         = r.forecast(model, h=48, level=80)
        vector = fcast.rx2('fitted')
        preds  = np.asarray(vector)
        
        # For some unknown reason, "forecast" is going way beyond the "h" periods I specify
        # So truncate the predictions at length of TestY
        length = len(panelDict["testY"])
        preds  = preds[:length].astype(int)
        
        
        rmse = calcRMSE(panelDict["testY"], preds)
        if rmse < bestRMSE:
            bestRMSE = rmse
            bestPreds = preds
    return bestPreds
    
def buildStl(panelDict, config):
    ''' STL is a univariate method so we skip the Xs and use on Y '''

    parms = getParms("STL")
    
    predsDF = process(panelDict, parms, config)
    return predsDF