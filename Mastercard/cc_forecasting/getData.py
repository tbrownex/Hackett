# -*- coding: utf-8 -*-
# 
"""
    Read the input file into a dataframe.
    
    - Rename columns
    - Allow a "Test" mode (set in "config") to read a smaller version of the larger dataset.
    - Convert the input "month" column to a standard python datetime
    - Assuming the first column will contain the date values
    - assuming the file will have the headers in the first row
"""

from dateutil import parser
import pandas as pd
import logging

__author__ = 'The Hackett Group'

def getData(config):
    
    #cols = ['Country', 'Program', 'Customer', 'Driver', 'Month', 'Amount']
    if config['Test']:
        # Read a sample number of drivers (tbd) or send to filter(df) to preselect conditional
        # arguments
        #df = pd.read_csv(config['baseDir'] + config['fileName'] + 'Egypt.csv', header=0, names=cols)
        print('Test Mode selected by config')
    else:
        df = pd.read_csv(config['baseDir'] + config['fileName'], header=0,\
                         dtype={'UUID': object},parse_dates=[0])
        
    return(df)
