''' Get the command line arguments:
    Mandatory
    - model: the type of forecasting model to run
    - trainInd: whether to train or use an existing model
    
    Optional arguments
    - log: the logging level (overrides "config")
    - normalize: the algorithm to use when normalizing the input'''
    
__author__ = "The Hackett Group"

import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("testInd", \
                        choices=['test','full'], \
                        help="Run against the full data or a test portion")
    return parser.parse_args()