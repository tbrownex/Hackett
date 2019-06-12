#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a python globals file that creates instances
of global variables for use where absolutely necessary
"""

import pandas as pd

def init():
    global dfMetrics
    dfMetrics = pd.DataFrame()
    
    global collabDict
    collabDict = {}
    
    global collabOutput
    collabOutput = {}
    
    