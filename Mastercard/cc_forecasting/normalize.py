# -*- coding: utf-8 -*-
"""
For each UUID, normalize the y column
using a z-score transformation
ddof is set to 1 in the std function.
"""

__author__ = "The Hackett Group"

import pandas as pd

def z_score(df,config):
    
    listDriver = df.Driver.unique()
    
    data = []
    for Driver in listDriver:
        driverDf = df.loc[df['Driver']==Driver]
        
        # Z Score transform each Driver
        df["z"] = (df["y"] - df["y"].mean())/df["y"].std(ddof=1)
    
    
    return df