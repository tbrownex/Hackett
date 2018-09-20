import numpy  as np
from rpy2.robjects import r
#from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

def process(model, config):
    forecast = importr('forecast')
    fcast    = r.forecast(model, h=config["testMonths"], level=80)
    vector   = fcast.rx2('fitted')
    preds    = np.asarray(vector)
    return preds.astype(int)