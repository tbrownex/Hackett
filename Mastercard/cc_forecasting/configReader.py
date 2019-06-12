# -*- coding: utf-8 -*-
"""
Config File Parser using config.ini
"""
import sys
import configparser

# These are local environment specific
sys.path.insert(0,'C:/Users/e084332/Documents/repos/sandbox/')
sys.path.insert(1,'C:/Users/e084332/Documents/repos/cc_forecasting/')
sys.path.insert(2,'/Users/wjc/Documents/repos/sandbox/')
sys.path.insert(3,'/Users/wjc/Documents/repos/cc_forecasting/')

def config():
    config = configparser.ConfigParser()
    config.read(sys.path[0] + 'config.ini')
    
    
    return config