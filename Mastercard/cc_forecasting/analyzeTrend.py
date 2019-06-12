# -*- coding: utf-8 -*-
"""
For each UUID compute a trend metric and write to output
"""

__author__ = "The Hackett Group"

import pandas as pd
import numpy as np

def trend(df,config):
    
    #Create empty lists each iteration
    uuid = []
    driver = []
    pct_change = []
    log_ret = []
        
    #listDriver = df.Driver.unique()
    listDriver = df.Driver.unique()
    
    for driverTemp in listDriver:
        # Create a temp df of each 'x' then iterate (keto, gluten, kombucha)
        driverDf = df[df['Driver'] == driverTemp].copy()
        uuid.append(str(driverDf.iloc[0,1]))
        
        # Replace zero with 1
        driverDf = driverDf.replace(0,1)
        
        # Add a 2 computed columns to the df 
        driverDf['pct_change'] = driverDf.y.pct_change()
        driverDf['log_ret'] = np.log(driverDf.y) - np.log(driverDf.y.shift(1))
        
        # Replace and drop Nan, Inf, -inf
        #driverDf = driverDf.replace([np.inf, -np.inf],np.nan).dropna(subset=['pct_change','log_ret'], how = 'all')
        
        # Sum the 2 computed columns and stick the result into the list objects
        driver.append(str(driverTemp))
        pct_change.append(driverDf['pct_change'].sum())
        log_ret.append(driverDf['log_ret'].sum())
              
    metric = ['trend'] *len(driver) 
    driverOutput = pd.DataFrame([uuid,driver,metric,log_ret]).T
        
        
    print(driverOutput)
    
    return df