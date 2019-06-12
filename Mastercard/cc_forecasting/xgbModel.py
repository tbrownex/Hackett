# -*- coding: utf-8 -*-
"""
Depreciated this module 10/27/2018
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

def trainXGBoost(df,config):
    
    train = df.loc['2004-01':'2017-07'].drop(['Driver'],axis=1)
    test = df.loc['2017-08':].drop(['Driver'],axis=1)

    # Just the X axis (index)
    X_train = train.index # Lagged features but I only want to provide dates
    y_train = train.dropna().y # Represents the original values
    X_test = test.index
    
    xgb = XGBRegressor()
    model = xgb.fit(X_train, y_train)
    
    # Much code to add here - see my notebook test file


    
    return df

