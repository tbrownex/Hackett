# -*- coding: utf-8 -*-
"""
Simple row inclusion filter for test purposes
Created on Thu Sep 25 11:48:52 2018

@author: will.cairns
"""

import pandas as pd
import numpy as np

def filterData(df, config):

    df = df.loc[df['Driver'] == config['filterDriver']]
    
    
    return(df)

