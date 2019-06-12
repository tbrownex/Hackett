# -*- coding: utf-8 -*-
"""
Eight number summary from dataframe
Created on Thu Sep 20 09:23:52 2018

@author: will.cairns
"""

import pandas as pd
import numpy as np
from metricsCollector import listener

def descript(df, config):

    listDriver = df.Driver.unique()
    dfOutput = pd.DataFrame()
    
    for Driver in listDriver: 
        driverDf = df.loc[df['Driver']==Driver]

        U = [] # UUID List
        D = [] # Driver List

        statsDict = dict(driverDf['y'].describe())

        S = list(statsDict.values()) # Statistics (8)
        M = list(statsDict.keys()) # Metric Names

        for i in range(len(statsDict.keys())):
            U.append(driverDf.iloc[0,2]) # UUID's
            D.append(Driver) # Driver Name

        dfTemp = pd.DataFrame([U,D,M,S]).T
        dfTemp.columns= ['UUID', 'Driver', 'Metric', 'Value']

        dfOutput = pd.concat([dfOutput,dfTemp],axis = 0)
    
    listener(dfOutput)
    
    return(df)
