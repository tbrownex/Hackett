# Use the Dickey Fuller test for stationarity
# CI is the confidence interval
# prt is the binary print flag

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def ADF(series, CI, prt):
    assert (CI in ["1%", "5%", "10%"]), "CI must be one of 1%, 5% or 10%"
    result = adfuller(series)
    ADFstat = result[0]
    Pvalue  = result[1]
    ciValue = result[4][CI]
    if ADFstat < ciValue:
        stationary = True
    else:
        stationary = False
    if prt:
        print('ADF Statistic: {:.3f}'.format(ADFstat))
        print('P-value: {:>15.6f}'.format(Pvalue))
    return stationary