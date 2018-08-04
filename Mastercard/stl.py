import pandas as pd
import numpy  as np
import rpy2 as r
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import matplotlib.pyplot as plt
import operator
from statsmodels.tsa.arima_model import ARIMA

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

def forecastARIMA(adjusted):
    model     = ARIMA(adjusted, order=(1, 1, 2))
    model_fit = model.fit(disp=0)
    return model_fit.forecast(steps=TEST_WEEKS)[0]

def printResults(forecast, test, train):
    fcastMAPE = getMAPE(forecast, test)
    baseline  = np.array([train.iloc[-1]]*TEST_WEEKS)
    baseMAPE  = getMAPE(baseline, test)
    print("MAPE: {:.2%}".format(fcastMAPE))
    print("MAPE: {:.2%}".format(baseMAPE))

if __name__ == "__main__":
    train, test = prepareData()
    seasonal    = getSeasonality(train)
    adjusted    = train - seasonal       # Remove the seasonal to get the adjusted
    forecast    = forecastARIMA(adjusted)
    # Ensure "forecast" and "test" have the same durations
    assert (forecast.shape[0] == test.shape[0])
    forecast    += seasonal[-52:(-52+TEST_WEEKS)]
    printResults(forecast, test, train)