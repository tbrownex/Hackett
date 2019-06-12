# -*- coding: utf-8 -*-
"""
For each UUID compute a slope metric and write to output
"""

__author__ = "The Hackett Group"

import pandas as pd
import numpy as np
from scipy.stats import linregress
from metricsCollector import listener

def slope(df,config):
    
    #Create empty lists
    uuid = []
    driver = []
    result = []
        
    #listDriver = df.Driver.unique()
    listDriver = df.Driver.unique()
    
    for driverTemp in listDriver:
        # Create a temp df of each 'x' then iterate (keto, gluten, kombucha)
        driverDf = df[df['Driver'] == driverTemp].copy()

        # Replace zero with 1
        driverDf = driverDf.replace(0,1)

        
        z = linregress(list(range(0,driverDf['y'].count())),driverDf['y'])
        uuid.append(str(driverDf.iloc[0,2]))
        driver.append(str(driverTemp))
        result.append(z.slope)            
        
    metric = ['Slope'] * len(driver)
    dfOutput = pd.DataFrame([uuid,driver,metric,result]).T
    
    dfOutput.columns = ['UUID','Driver', 'Metric', 'Value']

    listener(dfOutput)
    
    return df