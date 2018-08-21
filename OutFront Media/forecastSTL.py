import numpy  as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

# Input has X and Y values. STL is a univariate method, so we skip the Xs and use on Y 
def forecastSTL(dataDict):
    values = list(dataDict["trainY"])
    values = r.ts(values, frequency=24)
    
    model  = r.stl(values, s_window = 'periodic')
    
    forecast = importr('forecast')
    fcast    = r.forecast(model, h=48, level=80)
    vector   = fcast.rx2('fitted')
    preds    = np.asarray(vector)
    
    # For some unknown reason, "forecast" is going way beyond the "h" periods I specify
    # So truncate the predictions at length of TestY
    length = len(dataDict["testY"])
    preds  = preds[:length]
    
    return preds