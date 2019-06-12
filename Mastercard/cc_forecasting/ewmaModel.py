# -*- coding: utf-8 -*-
"""
Depreciated this module 10/27/2018
"""

import numpy as np
import pandas as pd
from abstractWriter import csvWriter
from getConfig import getConfig


def trainEWMA(df,config):

    dfOutput = pd.DataFrame()
    uuid = []
    driver = []
    mape = []
    monthlydiff = []

    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,decimals = 2)
    
    listDriver = df.Driver.unique()
    
    for driverTemp in listDriver:
        driverDf = df[df['Driver'] == driverTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        
                # First two columns for the output file
        uuid.append(str(driverDf.iloc[0,2]))
        driver.append(driverTemp)
    
        train = driverDf.loc['2004-01':'2017-07'].copy()
        test = driverDf.loc['2017-08':].copy()
        
        futureInterval = test['y'].count()
        
        train['ewma'] = train.y.ewm(halflife=2).mean()
        
        # Put the last N ewm values into a list
        trainEWM = list(train['ewma'].tail(futureInterval))

        predict = []
        i = 0
        while i < futureInterval:
          predict.append(np.mean(trainEWM[i:]))
          i += 1
  
        test['yhat'] = predict
        
        mape.append(mean_absolute_percentage_error(test.y.values,test.yhat.values))
        
        monthlydiff.append((abs(np.sum(test.y.values) - np.sum(test.yhat.values))) / futureInterval)
        
    dfOutput = pd.DataFrame([uuid,driver,mape,monthlydiff]).T
    dfOutput.columns = ['UUID', 'Driver','MAPE','Diff_LC']
    csvWriter(dfOutput,'ewmaModel',config)

    return df