# -*- coding: utf-8 -*-
"""
Seasonal ARIMA from predefined inputs
Modification to predict training and test data
and send all to the output file
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings
import time
import logging

log = logging.getLogger(__name__)

from abstractWriter import csvWriter
from abstractCollaborator import abstractCollector
from getConfig import getConfig
from abstractPostprocessor import normalizeAdjustment
from abstractPostprocessor import nominalAdjustment
import settings

"""
Helper Function to read ARIMA parameters file
the location and filename are set via the config
data is read in as tuples and passed back to the
calling function
"""
import numpy as np
import pandas as pd
import ast

def getParams(config):
    
        params = pd.read_csv(config['paramDir'] + config['paramFile'], header=0,\
            sep='|',converters={"pdq": ast.literal_eval,"PDQs": ast.literal_eval})
        
        return(params)


def trainArima(df,config):

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    start_time = time.time()
    
    # Best Parameters for Processed Volume from Model training in Alteryx
    params = getParams(config)
    pdq = list(params['pdq'])
    seasonal_pdq = list(params['PDQs'])
    
    dfOutput = pd.DataFrame()
    listDriver = df.Driver.unique()

    for driverTemp in listDriver:
        driverDf = df[df['Driver'] == driverTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        driverDf.sort_index(inplace=True)
        driverDf['increment'] = list(range(driverDf.shape[0]))
        driverDf['windowId'] = driverDf.apply(lambda x: 'train' if x.Date <=  \
                               driverDf.iloc[driverDf.shape[0] - int(config['inSamplePeriods'])-1:].Date[0] \
                               else 'test',axis=1)

        best_pdq, best_seasonal, best_score = None, None, float("inf")

        for param,seasonalparam in zip(pdq, seasonal_pdq):
            try:
                mod = SARIMAX(driverDf.loc[driverDf['windowId']=='train',['y']],
                              order=param,
                              seasonal_order=seasonalparam,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit()
                # Test to collect the best fitting model
                if res.aic < best_score:
                    best_pdq, best_seasonal, best_score = param, seasonalparam, res.aic
                    #print('ARIMA{}x{} - AIC:{}'.format(param, seasonalParam, res.aic))
            except:
                continue

        mod = SARIMAX(driverDf.loc[driverDf['windowId']=='train',['y']],
                      order=best_pdq,
                      seasonal_order=best_seasonal,
                      enforce_stationarity=False,
                      enforce_invertibility=False)

        res = mod.fit()

        start_index = driverDf.increment.min() #test.index.min()
        end_index = driverDf.increment.max()

        driverDf['yhat'] = res.predict(start_index,end_index,typ='levels')

        # Calculated Fields
        driverDf['mape'] = driverDf.apply(lambda x: np.round(np.mean(np.abs((x.y - x.yhat) / x.y)),decimals = 6),axis=1)
        driverDf['variance_abs'] = driverDf.apply(lambda x: abs(x.y - x.yhat),axis=1)
        driverDf['modelTyp'] = 'arima'

        dfOutput = pd.concat([dfOutput,driverDf], axis=0)
    
    dfOutput = dfOutput[['Date','UUID','Driver','y','yhat','mape','variance_abs','windowId','modelTyp']]
    csvWriter(dfOutput,'trainARIMA',config)
    abstractCollector(dfOutput,'arima',config)
    
    log.info('Rows in this output data: %s', dfOutput.shape[0])
    
    return df


def predictArima(df,config):

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    
    # Best Parameters for Processed Volume from Model training in Alteryx
    params = getParams(config)
    pdq = list(params['pdq'])
    seasonal_pdq = list(params['PDQs'])

    dfOutput = pd.DataFrame()

    # This code filters the collabOutput dict to only those UUIDs where Arima
    # has returned the best MAPE on training data. The dataframe is then filtered
    # so it will only contain UUIDs from the list.
    uuidArima = []
    for uuid, modelInfo in settings.collabOutput.items():
        for key in modelInfo:
            if key == 'arima':
                uuidArima.append(uuid)  

    for uuidTemp in uuidArima:
        driverDf = df[df['UUID'] == uuidTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        driverDf.sort_index(inplace=True)
        driverDf['windowId'] = driverDf.apply(lambda x: 'train' if x.Date <=  \
                               driverDf.iloc[driverDf.shape[0] - int(config['inSamplePeriods'])-1:].Date[0] \
                               else 'test',axis=1)
        
        idx = pd.date_range(driverDf['Date'].min(), driverDf['Date'].max() + pd.offsets.MonthBegin(int(config['outSamplePeriods'])),freq='MS')
        driverDf = driverDf.reindex(idx,fill_value=np.nan)

        driverDf['increment'] = list(range(driverDf.shape[0]))
        driverDf['windowId'] = driverDf['windowId'].fillna('predict')
        driverDf['Date'] = driverDf.index
        driverDf['UUID'] = driverDf['UUID'][0]
        driverDf['Driver'] = driverDf['Driver'][0]
        
        best_pdq, best_seasonal, best_score = None, None, float("inf")

        for param,seasonalparam in zip(pdq, seasonal_pdq):
            try:
                mod = SARIMAX(driverDf.dropna().y,
                              order=param,
                              seasonal_order=seasonalparam,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit()
                # Test to collect the best fitting model
                if res.aic < best_score:
                    best_pdq, best_seasonal, best_score = param, seasonalparam, res.aic
                    #print('ARIMA{}x{} - AIC:{}'.format(param, seasonalParam, res.aic))
            except:
                continue
        #print('Best ARIMA = ', best_pdq,best_seasonal, 'Best AIC = ',best_score)

        mod = SARIMAX(driverDf.dropna().y,
                      order=best_pdq,
                      seasonal_order=best_seasonal,
                      enforce_stationarity=False,
                      enforce_invertibility=False)

        res = mod.fit()

        start_index = driverDf.increment.min() #test.index.min()
        end_index = driverDf.increment.max()

        driverDf['yhat'] = res.predict(start_index,end_index,typ='levels')

        # Calculated Fields
        # Calculated Fields
        driverDf['mape'] = driverDf.loc[driverDf['windowId']=='test'].apply(lambda x: np.round(np.mean(np.abs((x.y - x.yhat) / x.y)),decimals = 6),axis=1)
        driverDf['variance_abs'] = driverDf.loc[driverDf['windowId']=='test'].apply(lambda x: abs(x.y - x.yhat),axis=1)
        driverDf['modelTyp'] = 'arima'

        dfOutput = pd.concat([dfOutput,driverDf], axis=0)
    

    dfOutput = normalizeAdjustment(dfOutput)
    dfOutput = nominalAdjustment(dfOutput)
    dfOutput = dfOutput[['Date','UUID','Driver','y','yhat','yNormal','yhatNormal','yNominal','yhatNominal','mape','variance_abs','windowId','modelTyp']]
    
    csvWriter(dfOutput,'predictARIMA',config)
    
    log.info('Rows in this output data: %s', dfOutput.shape[0])

    return df