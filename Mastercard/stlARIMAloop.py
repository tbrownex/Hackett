''' Using JLP data for testing:
1) get the Seasonality of the data using "stl" function in R
2) adjust the actual sales data by this seasonality
3) run ARIMA against the resulting adjusted time series
4) make a forecast
5) compare the forecast to a test set

There's a "grid search" type of function - that's what the loop is for - to find
the best ARIMA parameters'''

import pandas as pd
import numpy  as np
import rpy2 as r
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import matplotlib.pyplot as plt
import operator
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")

PATH = "/home/tom/"
FILE = "store_sales_by_OLG-pre.csv"   # This is JLP data
TEST_WEEKS = 26

# "series" has an index which needs to be ignored (series.values)
def decompose(series, frequency, s_window, log=False,  **kwargs):
    series = series.values
    df = pd.DataFrame()
    if log: series = series.pipe(np.log)
    s = [x for x in series]
    length = len(series)
    s = r.ts(s, frequency=frequency)
    decomposed = [x for x in r.stl(s, s_window, robust=True).rx2('time.series')]
    df['observed']  = series
    df['trend']     = decomposed[length:2*length]
    df['seasonal']  = decomposed[0:length]
    df['residuals'] = decomposed[2*length:3*length]
    return df

# Get the MAPE between the inputs
def getMAPE(compare, test):
    diff = np.abs(compare-test)
    MAPE = diff.values/test
    return MAPE.mean()

def prepareData():
    # Use JLP sales data
    df = pd.read_csv(PATH+FILE, names=["store", "week", "sales"], header=0)
    # Aggregate the data so it's in format [Week, sales]
    sales = df.groupby(["week"])["sales"].sum()
    # Something weird happened the last 6 weeks so discard
    sales = sales[:-6]
    # Get rid of "2017(03)" format as the index and use sequence
    idx = [x for x in range(len(sales))]
    sales.index=idx
    # Create Train and Test sets
    train = sales[:-TEST_WEEKS]
    test  = sales[-TEST_WEEKS:]
    # Make sure splitting train & test didn't lose any data
    assert (sales.sum() - train.sum() - test.sum() < 0.1 )
    return train, test

# Run the R function "stl" to get the Seasonal component
def getSeasonality(train):
    tmpDF    = decompose(train, frequency=52, s_window="periodic")
    return np.array(tmpDF["seasonal"])

def forecastARIMA(adjusted, pdq):
    p = pdq[0]
    d = pdq[1]
    q = pdq[2]
    model     = ARIMA(adjusted, order=(p,d,q))
    model_fit = model.fit(disp=0)
    return model_fit.forecast(steps=TEST_WEEKS)[0]

if __name__ == "__main__":
    train, test = prepareData()
    seasonal    = getSeasonality(train)
    adjusted    = train - seasonal       # Remove the seasonal to get the adjusted
    bestMAPE = float("inf")
    for p in range(13):
        for d in range(3):
            for q in range(3):
                try:
                    forecast = forecastARIMA(adjusted, (p,d,q))
                    # Ensure "forecast" and "test" have the same durations
                    assert (forecast.shape[0] == test.shape[0])
                    forecast += seasonal[-52:(-52+TEST_WEEKS)]
                    score = getMAPE(forecast, test)
                    if score < bestMAPE:
                        bestMAPE = score
                        bestParms = (p,d,q)
                except:
                    pass
    print("Best performance was with MAPE: {:.2%} and parms: {}".format(bestMAPE, bestParms))
    baseline = [train.values[-1] for x in range(TEST_WEEKS)]
    score = getMAPE(baseline, test)
    print("Versus baseline of {:.2%}".format(score))