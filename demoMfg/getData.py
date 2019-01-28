from dateutil import parser
import pandas as pd
import logging

__author__ = "Tom Browne"

def getData(config, sep=","):
    ''' Read the input file into a dataframe'''
    train = pd.read_csv(config["dataLoc"]+config["fileName"], sep=sep)
    test  = pd.read_csv(config["dataLoc"]+config["testFile"], sep=sep)
    return train, test