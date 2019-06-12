# -*- coding: utf-8 -*-
"""
Abstract Post - Processing Functions
For Calendar Adjustments
"""

__author__ = "The Hackett Group"

import pandas as pd
import numpy as np
from calendar import monthrange, weekday
from datetime import date,datetime
import sys
import configparser
from configReader import config
conf = config()

print(sys.path[0])


conf.read(sys.path[0] + '/config.ini')

def normalizeAdjustment(df):
    
    """
    -Normalizer will take the completed forecast and cast the data
     into processing days.
    - Step 1 get called by the predict function & passed the output DataFrame
    - Step 2 read the config parameters for weighted processing days
    - Step 3 break the input dataframe into Transactions and Volume
    - Step 4 Using the weighting factor from the config, create a dictionary 
      object of month dates (keys) and processing days (values).
    - Step 5 add calculated columns to the TRX and VOL dataframes that are
      the Normalized values for y and yhat
    - Step 6 reassemble (concat) the TRX and VOL dataframes and pass back to the 
      calling predict function 
    """


    # Read the config region section and choose the appropriate parameters
    region = {k[4:]:v for k,v in dict(conf['region']).items() if k.startswith('mea.')}

    trxFactor = float(region.get('trxweight')) # Transaction Drivers
    volFactor = float(region.get('volweight')) # Volume Drivers
    
    #trxFactor = 0.5 # Transaction Drivers
    #volFactor = 0.7 # Volume Drivers
    
    trxDict = {}
    volDict = {}
    
    datesUniq = pd.DatetimeIndex(df.Date.unique())
    
    for dt in datesUniq:
        startdate = dt
        mondayCount = 0
        sundayCount = 0

        daysInMonth = range(1,monthrange(startdate.year,startdate.month)[1]+1)

        for day in daysInMonth:
            if weekday(startdate.year, startdate.month, day) == 0:
                #print('Year: ' + str(startdate.year) + ' Month: ' + str(startdate.month) + ' Day ' + str(day) + ' is a Monday')
                mondayCount +=1
            if weekday(startdate.year, startdate.month, day) == 6:
                sundayCount +=1
            
        processingDayCount = (monthrange(startdate.year,startdate.month)[1]) - sundayCount
        trxProcessingDayCount = processingDayCount + (mondayCount * trxFactor)
        volProcessingDayCount = processingDayCount + (mondayCount * volFactor)

        trxDict[dt] = trxProcessingDayCount
        volDict[dt] = volProcessingDayCount
        
    #print(volDict)
        
    # Make a Volume only DataFrame and a Transactions only DataFrame
    dfTrx = df[df['Driver'].astype(str).str.contains('Transact')]
    dfVol = df[df['Driver'].astype(str).str.contains('Volume')]
    
    if dfVol.shape[0] > 0:
        dfVol['procDaysNormal'] = dfVol['Date'].map(volDict)
        dfVol['yNormal'] = dfVol.apply(lambda x: np.nan if pd.isnull(x['y']) else x['procDaysNormal'] * (x['y']/30.5),axis=1)
        dfVol['yhatNormal'] = dfVol.apply(lambda x: np.nan if pd.isnull(x['yhat']) else x['procDaysNormal'] * (x['yhat']/30.5),axis=1)
    
    if dfTrx.shape[0] > 0:
        dfTrx['procDaysNormal'] = dfTrx['Date'].map(trxDict)
        dfTrx['yNormal'] = dfTrx.apply(lambda x: np.nan if pd.isnull(x['y']) else x['procDaysNormal'] * (x['y']/30.5),axis=1)
        dfTrx['yhatNormal'] = dfTrx.apply(lambda x: np.nan if pd.isnull(x['yhat']) else x['procDaysNormal'] * (x['yhat']/30.5),axis=1)
    
    df = pd.concat([dfVol,dfTrx],axis=0,sort=False)
    
    return df

def nominalAdjustment(df):

    """
    -Norminal will take the completed forecast and cast the data
     into calendar days.
    - Step 1 get called by the predict function
    - Step 2 Using the date values from the input data, create a dictionary 
      object of month dates (keys) and calendar days (values).
    - Step 3 add calculated columns to the dataframe that is
      the Norminal values for y and yhat
    - Step 4 Pass back to the calling predict function 
    """
    monthDict = {}
    
    datesUniq = pd.DatetimeIndex(df.Date.unique())
    
    for dt in datesUniq:
        startdate = dt

        daysInMonth = monthrange(startdate.year,startdate.month)[1]
        #print('Date: ' + str(dt) + ' Days in Month: ' + str(daysInMonth))
        monthDict[dt] = daysInMonth
        
    df['procDaysNominal'] = df['Date'].map(monthDict)
    df['yNominal'] = df.apply(lambda x: np.nan if pd.isnull(x['y']) else x['procDaysNominal'] * (x['y']/30.5),axis=1)
    df['yhatNominal'] = df.apply(lambda x: np.nan if pd.isnull(x['yhat']) else x['procDaysNominal'] * (x['yhat']/30.5),axis=1)
            
    return df
