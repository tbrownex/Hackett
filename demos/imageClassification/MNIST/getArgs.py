''' Get the command line arguments:
    Mandatory
    - None
    
    Optional arguments
    - log: the logging level (overrides "config")
    - normalize: the algorithm to use when normalizing the input'''
    
__author__ = "The Hackett Group"

import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    '''parser.add_argument("genFeatures", \
                        choices=['Y','N'], \
                        help="Generate additional features or not")
    parser.add_argument("-O", "--Outliers", \
                        choices=['Y','N'], \
                        help="Remove outliers or not; Y to remove")'''
    return parser.parse_args()