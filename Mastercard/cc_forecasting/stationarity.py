# -*- coding: utf-8 -*-
"""
Test for stationarity
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from timestampGenerator import tstamp
from metricsCollector import listener

def dickey_fuller(df, config):

    listDriver = df.Driver.unique()
    dfOutput = pd.DataFrame()
    
    for Driver in listDriver:
        driverDf = df.loc[df['Driver']==Driver]

        U = [] # UUID List
        D = [] # Driver List
        M = [] # Metrics
        S = [] # Statistics

        dftest = adfuller(driverDf.y.values, autolag='AIC')

        output = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
        for key,value in dftest[4].items():
            output['Critical Value (%s)'%key] = value

        S=(list(output))
        M=(list(output.index))

        for i in range(len(S)):
            U.append(driverDf.iloc[0,2]) # UUID's
            D.append(Driver) # Driver Name

        dfTemp = pd.DataFrame([U,D,M,S]).T
        dfTemp.columns= ['UUID', 'Driver', 'Metric', 'Value']

        dfOutput = pd.concat([dfOutput,dfTemp],axis = 0)

    listener(dfOutput)

    return(df)
