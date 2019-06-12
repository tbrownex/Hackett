# -*- coding: utf-8 -*-
"""
ETS Model
"""
import numpy as np
import pandas as pd
import csv
import warnings
import logging
from datetime import datetime,date

log = logging.getLogger(__name__)

import settings
from abstractWriter import csvWriter
from abstractCollaborator import abstractCollector
from abstractPostprocessor import normalizeAdjustment
from abstractPostprocessor import nominalAdjustment
from getConfig import getConfig
import math
from statsmodels.tsa.api import ExponentialSmoothing

def trainETS(df,config):
    dfOutput = pd.DataFrame()    
    listDriver = df.Driver.unique()
    
    for driverTemp in listDriver:
        
        driverDf = df[df['Driver'] == driverTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        driverDf.sort_index(inplace=True)
        driverDf.index.freq = 'MS'

        driverDf['increment'] = list(range(driverDf.shape[0]))
        driverDf['windowId'] = driverDf.apply(lambda x: 'train' if x.Date <=  \
                               driverDf.iloc[driverDf.shape[0] - int(config['inSamplePeriods'])-1:].Date[0] \
                               else 'test',axis=1)

        # parameters
        alpha = 0.4
        beta = 0.2
        gamma = 0.01
        phi = 12

        # initialise model
        ets_model = ExponentialSmoothing(driverDf.loc[driverDf['windowId']=='train',['y']],\
                                         trend='add', seasonal='add', 
                                         seasonal_periods=phi)

        ets_fit = ets_model.fit(smoothing_level=alpha, smoothing_slope=beta,
        smoothing_seasonal=gamma)

        # forecast train and test
        driverDf['yhat'] = ets_fit.predict(start=0,end=(driverDf.shape[0]-1)).values

        # Calculated Fields
        driverDf['mape'] = driverDf.apply(lambda x: np.round(np.mean(np.abs((x.y - x.yhat) / x.y)),decimals = 6),axis=1)
        driverDf['variance_abs'] = driverDf.apply(lambda x: abs(x.y - x.yhat),axis=1)
        driverDf['modelTyp'] = 'ETS'

        dfOutput = pd.concat([dfOutput,driverDf], axis=0)
    
    dfOutput = dfOutput[['Date','UUID','Driver','y','yhat','mape','variance_abs','windowId','modelTyp']]
    csvWriter(dfOutput,'trainETS',config)
    abstractCollector(dfOutput,'ets',config)
    
    log.info('Rows in this output data: %s', dfOutput.shape[0])
    
    return df


def predictETS(df,config):
    dfOutput = pd.DataFrame()

    # This code filters the collabOutput dict to only those UUIDs where ETS
    # has returned the best MAPE on training data. The dataframe is then filtered
    # so it will only contain UUIDs from the list.
    uuidETS = []
    for uuid, modelInfo in settings.collabOutput.items():
        for key in modelInfo:
            if key == 'ets':
                uuidETS.append(uuid)
    
    for uuidTemp in uuidETS:
        
        driverDf = df[df['UUID'] == uuidTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        driverDf.sort_index(inplace=True)
        driverDf.index.freq = 'MS'

        driverDf['increment'] = list(range(driverDf.shape[0]))
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

        # parameters
        alpha = 0.4
        beta = 0.2
        gamma = 0.01
        phi = 12

        # initialise model
        ets_model = ExponentialSmoothing(driverDf.dropna().y,\
                                         trend='add', seasonal='add', 
                                         seasonal_periods=phi)

        ets_fit = ets_model.fit(smoothing_level=alpha, smoothing_slope=beta,
        smoothing_seasonal=gamma)

        # forecast in-sample and out-sample
        driverDf['yhat'] = ets_fit.predict(start=0,end=(driverDf.shape[0]-1)).values

        # Calculated Fields
        driverDf['mape'] = driverDf.loc[driverDf['windowId']=='test'].apply(lambda x: np.round(np.mean(np.abs((x.y - x.yhat) / x.y)),decimals = 6),axis=1)
        driverDf['variance_abs'] = driverDf.loc[driverDf['windowId']=='test'].apply(lambda x: abs(x.y - x.yhat),axis=1)
        driverDf['modelTyp'] = 'ets'

        dfOutput = pd.concat([dfOutput,driverDf], axis=0)
    
    dfOutput = normalizeAdjustment(dfOutput)
    dfOutput = nominalAdjustment(dfOutput)
    dfOutput = dfOutput[['Date','UUID','Driver','y','yhat','yNormal','yhatNormal','yNominal','yhatNominal','mape','variance_abs','windowId','modelTyp']]
    csvWriter(dfOutput,'predictETS',config)
    
    log.info('Rows in this output data: %s', dfOutput.shape[0])
    
    return df