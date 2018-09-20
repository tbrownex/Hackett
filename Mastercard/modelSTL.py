import numpy  as np
import pandas as pd
from rpy2.robjects import r
#from rpy2.robjects import r, pandas2ri
#from rpy2.robjects.packages import importr

# TODO
# add Reported with frequency "Quarterly"

def process(train):
    values = list(train["Amount"])
    values = r.ts(values, frequency=12)
    return r.stl(values, s_window = 'periodic')