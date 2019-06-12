''' Get the command line arguments:
    Mandatory
    - model: the type of forecasting model to run
    - trainInd: whether to train or use an existing model
    - Added constants in each argument
    
    Optional arguments
    - log: the logging level (overrides "config")
    - normalize: the algorithm to use when normalizing the input'''
    
__author__ = "The Hackett Group"

import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", \
                        help="set logging level",
                        nargs='?',
                        const='Error')
    parser.add_argument("-n", "--normalize", \
                        choices=['zadj','minmax','log'], \
                        help="Zscore the input",
                        nargs='?',
                        const='zadj')
    parser.add_argument("model", \
                        choices=['arima','stl','xgb'], \
                        help="forecast model type",
                        nargs='?',
                        const='arima')
    parser.add_argument("trainInd", \
                        choices=['train','infer'], \
                        help="Train or run the model",
                        nargs='?',
                        const='train')
    return parser.parse_args()