''' Read the input file into a dataframe.
    
    - Rename columns
    - Allow a "Test" mode (set in "config") to read a smaller version of the larger dataset.
    - Convert the input "month" column to a standard python datetime'''

from dateutil import parser
import pandas as pd
import logging

__author__ = "The Hackett Group"

def getData(config, args):
    df = pd.read_csv(config["dataLoc"]+config["inputFile"])
    
    if args.testInd == "test":
        df = df.sample(frac=0.2)
    
    return df