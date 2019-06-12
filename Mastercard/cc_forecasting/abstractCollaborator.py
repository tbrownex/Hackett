# -*- coding: utf-8 -*-
"""
For each Model, add the UUID and MAPE to a dict
which gets passed to the abstractCollaborator
for parsing before the predictions are run
"""

__author__ = "The Hackett Group"

import pandas as pd
import numpy as np
import settings
from getConfig  import getConfig

def abstractCollector(df,modelType,config):
    
    """
    -Collector receives a dataframe from each model training
    -Collector creates an aggregate dataframe groupby UUID
    -Collector populates (updates or appends) a global dict object 
    - with UUID as keys and nested dict containing the model and mape values
    -Collectors output dict is read by bestModel function
    """

    dfCollab = df.loc[df['windowId']=='test'].groupby\
                           (['UUID'])['mape'].mean().reset_index()

    for row in dfCollab.iterrows():
        if row[1]['UUID'] in settings.collabDict:
            # Update the existing UUID key with a model and mape
            settings.collabDict[row[1]['UUID']].update({modelType: row[1]['mape']})
        else:
            # Add a new key and value to the dict
            settings.collabDict[row[1]['UUID']] = {modelType: row[1]['mape']}
            
    return

def bestModel(df,config):
    
    """
    - Called from Main, this function inherits the df but doesn't use it.
    - The only job for this function is to parse the dict created by 
    - abstractCollector and filter down to a new dict having a single model
    - and mape from model training. This in turn gets passed/read by model
    - predictors (predictArima, predictBsts, predictSVR) and used to identify 
    - which drivers will be forecasted.
    """
    
    for uuid, modelInfo in settings.collabDict.items():
        bestModel,bestMape = None, float("inf")
        for key in modelInfo:
            if modelInfo[key] < bestMape:
                bestModel, bestMape = key, modelInfo[key]
            settings.collabOutput[uuid] = {bestModel:bestMape}
            
    return df