import numpy  as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

def decompose(model, length):
    matrix = model.rx2('time.series')
    decomp = [x for x in matrix]
    
    df              = pd.DataFrame()
    df['trend']     = decomp[length:2*length]
    df['seasonal']  = decomp[0:length]
    df['residuals'] = decomp[2*length:3*length]
    return df

# Input has X and Y values. STL is a univariate method, so we skip the Xs and use on Y 
def forecastSTL(dataDict):
    values = list(dataDict["trainY"])
    length = len(values)
    values = r.ts(values, frequency=24)
    
    model  = r.stl(values, s_window = 'periodic')
    
    df = decompose(model, length)
    df["actual"] = dataDict["trainY"].values
    df.to_csv("/home/tbrownex/"+"STLmodel.csv", index=False)
    
    forecast      = importr('forecast')
    forecastCount = len(dataDict["testX"])
    fcast         = r.forecast(model, h=48, level=80)
    vector = fcast.rx2('fitted')
    preds  = np.asarray(vector)
    
    df = pd.DataFrame()
    df["actual"] = dataDict["testY"].values
    rows = len(dataDict["testY"].values)
    df["pred"] = preds[:rows]
    df.to_csv("/home/tbrownex/"+"STLpredictions.csv", index=False)
    input()
    return preds