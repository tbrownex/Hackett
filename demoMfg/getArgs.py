''' Get the command line arguments:
    Mandatory
    - genFeature: whether to use a generated feature or not
    
    Optional arguments
    - log: the logging level (overrides "config")
    - normalize: the algorithm to use when normalizing the input'''
    
__author__ = "The Hackett Group"

import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("genFeature", \
                        choices=['Y','N'], \
                        help="Use the moving standard deviation or not")
    return parser.parse_args()