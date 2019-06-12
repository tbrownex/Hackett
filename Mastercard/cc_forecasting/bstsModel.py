# -*- coding: utf-8 -*-
"""
Bayesian Structural Time Series (FB Prophet)
"""

import numpy as np
import pandas as pd
import csv
import warnings
import logging
from scipy.stats import zscore
from datetime import datetime,date
from fbprophet import Prophet

log = logging.getLogger(__name__)


from abstractWriter import csvWriter
from abstractCollaborator import abstractCollector
from abstractPostprocessor import normalizeAdjustment
from abstractPostprocessor import nominalAdjustment
from getConfig import getConfig
import settings


def trainBSTS(df,config):
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    
    dfOutput = pd.DataFrame()    
    listDriver = df.Driver.unique()
    
    for driverTemp in listDriver:
        prediction = {}
        
        driverDf = df[df['Driver'] == driverTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        driverDf.sort_index(inplace=True)
        driverDf = driverDf.rename(columns={'Date':'ds'})
        #driverDf['windowId'] = driverDf.apply(lambda x: 'train' if x.ds <= pd.to_datetime(['20170701']) else 'test',axis=1)
        driverDf['windowId'] = driverDf.apply(lambda x: 'train' if x.ds <=  \
                driverDf.iloc[driverDf.shape[0] - int(config['inSamplePeriods'])-1:].ds[0] \
                else 'test',axis=1)
        
        # ----------- Rolling Mean Imputation -----------------
        driverDf['zscore'] = np.abs(zscore(driverDf['y'].values))
        driverDf['rolling'] = driverDf['y'].shift().rolling(13,min_periods=1).mean()
        driverDf['y_adj'] = np.where(driverDf['zscore']<2.25,driverDf['y'],driverDf['rolling'])
        driverDf = driverDf.rename(columns={'y': 'y_actual'})
        driverDf = driverDf.rename(columns={'y_adj': 'y'})
        # ----------------- Done -------------------------------
        
        # ------------- Build a DataFrame of Ramadan Holidays -------------------------
        # ------------- 3 Weeks after start with -1,0 month window ----------------------
        holidayDf = pd.DataFrame({'holiday': 'Ramadan','ds': \
        pd.to_datetime(['12/1/2000', '12/1/2001', '11/1/2002',\
        '11/1/2003', '11/1/2004', '10/1/2005',\
        '10/1/2006', '10/1/2007', '09/1/2008',\
        '09/1/2009', '08/1/2010', '08/1/2011',\
        '08/1/2012', '07/1/2013', '07/1/2014',\
        '07/1/2015', '07/1/2016', '06/1/2017',\
        '06/1/2018', '05/1/2019', '05/1/2020',\
        '05/1/2021', '04/1/2022', '06/1/2023']),\
        'lower_window': -1,'upper_window': 0})

        bstsModel = Prophet(holidays=holidayDf)
        bstsModel.fit(driverDf.loc[driverDf['windowId']=='train'])
    
        futureDates = bstsModel.make_future_dataframe( \
                      periods=driverDf.loc[driverDf['windowId']=='test'].shape[0], \
                      freq='MS')
             
        bstsForecast = bstsModel.predict(futureDates)
        prediction[driverDf.Driver[0]] = bstsForecast
        
        # Return is a dict
        tempOutput = pd.DataFrame.from_dict(bstsForecast)
        driverDf['yhat'] = tempOutput['yhat'].values

        # Calculated Fields
        driverDf['mape'] = driverDf.apply(lambda x: np.round(np.mean(np.abs((x.y - x.yhat) / x.y)),decimals = 6),axis=1)
        driverDf['variance_abs'] = driverDf.apply(lambda x: abs(x.y - x.yhat),axis=1)
        driverDf['modelTyp'] = 'bsts'

        dfOutput = pd.concat([dfOutput,driverDf], axis=0)
    
    dfOutput = dfOutput[['ds','UUID','Driver','y','yhat','mape','variance_abs','windowId','modelTyp']]
    csvWriter(dfOutput,'trainBSTS',config)
    abstractCollector(dfOutput,'bsts',config)
    
    log.info('Rows in this output data: %s', dfOutput.shape[0])
    
    return df


def predictBSTS(df,config):
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    
    dfOutput = pd.DataFrame()    
    
    # This code filters the collabOutput dict to only those UUIDs where BSTS
    # has returned the best MAPE on training data. The dataframe is then filtered
    # so it will only contain UUIDs from the list.
    uuidBSTS = []
    for uuid, modelInfo in settings.collabOutput.items():
        for key in modelInfo:
            if key == 'bsts':
                uuidBSTS.append(uuid)
    
    for uuidTemp in uuidBSTS:
        prediction = {}
        
        driverDf = df[df['UUID'] == uuidTemp].copy()
        driverDf.set_index('Date',inplace=True,drop=False)
        driverDf.sort_index(inplace=True)
        driverDf = driverDf.rename(columns={'Date':'ds'})
        driverDf['windowId'] = driverDf.apply(lambda x: 'train' if x.ds <=  \
                               driverDf.iloc[driverDf.shape[0] - int(config['inSamplePeriods'])-1:].ds[0] \
                               else 'test',axis=1)
        
        # ----------- Rolling Mean Imputation -----------------
        driverDf['zscore'] = np.abs(zscore(driverDf['y'].values))
        driverDf['rolling'] = driverDf['y'].shift().rolling(13,min_periods=1).mean()
        driverDf['y_adj'] = np.where(driverDf['zscore']<2.25,driverDf['y'],driverDf['rolling'])
        driverDf = driverDf.rename(columns={'y': 'y_actual'})
        driverDf = driverDf.rename(columns={'y_adj': 'y'})
        # ----------------- Done -------------------------------
        
        # ------------- Build a DataFrame of Ramadan Holidays -------------------------
        # ------------- 3 Weeks after start with -1,0 month window ----------------------
        holidayDf = pd.DataFrame({'holiday': 'Ramadan','ds': \
        pd.to_datetime(['12/1/2000', '12/1/2001', '11/1/2002',\
        '11/1/2003', '11/1/2004', '10/1/2005',\
        '10/1/2006', '10/1/2007', '09/1/2008',\
        '09/1/2009', '08/1/2010', '08/1/2011',\
        '08/1/2012', '07/1/2013', '07/1/2014',\
        '07/1/2015', '07/1/2016', '06/1/2017',\
        '06/1/2018', '05/1/2019', '05/1/2020',\
        '05/1/2021', '04/1/2022', '06/1/2023']),\
        'lower_window': -1,'upper_window': 0})

        bstsModel = Prophet(holidays=holidayDf)
        bstsModel.fit(driverDf)
    
        futureDates = bstsModel.make_future_dataframe( \
                      periods=int(config['outSamplePeriods']), \
                      freq='MS')
             
        bstsForecast = bstsModel.predict(futureDates)
        prediction[driverDf.Driver[0]] = bstsForecast
        
        # Return is a dict
        tempOutput = pd.DataFrame.from_dict(bstsForecast)
        tempOutput.set_index('ds',inplace=True,drop=False)
        driverDf = driverDf.reindex(tempOutput.index,fill_value=np.nan)

        driverDf['yhat'] = tempOutput['yhat'].values
        driverDf['Date'] = tempOutput['ds'].values
        driverDf['UUID'] = driverDf['UUID'][0]
        driverDf['Driver'] = driverDf['Driver'][0]
        driverDf['windowId'] = driverDf['windowId'].fillna('predict')

        # Calculated Fields
        driverDf['mape'] = driverDf.loc[driverDf['windowId']=='test'].apply(lambda x: np.round(np.mean(np.abs((x.y - x.yhat) / x.y)),decimals = 6),axis=1)
        driverDf['variance_abs'] = driverDf.loc[driverDf['windowId']=='test'].apply(lambda x: abs(x.y - x.yhat),axis=1)
        driverDf['modelTyp'] = 'bsts'

        dfOutput = pd.concat([dfOutput,driverDf], axis=0)
    
    dfOutput = normalizeAdjustment(dfOutput)
    dfOutput = nominalAdjustment(dfOutput)
    dfOutput = dfOutput[['Date','UUID','Driver','y','yhat','yNormal','yhatNormal','yNominal','yhatNominal','mape','variance_abs','windowId','modelTyp']]
    
    csvWriter(dfOutput,'predictBSTS',config)
    
    log.info('Rows in this output data: %s', dfOutput.shape[0])
    
    return df
