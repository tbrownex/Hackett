''' Read the input file into a dataframe.
    
    - Rename columns
    - Allow a "Test" mode (set in "config") to read a smaller version of the larger dataset.
    - Convert the input "month" column to a standard python datetime'''

from dateutil import parser
import pandas as pd
import logging

__author__ = "The Hackett Group"

def getData(config):
    
    cols = ["Country", "Program", "Customer", "Driver", "Month", "Amount"]
    if config["Test"]:
        df = pd.read_csv(config["dataLoc"]+"Egypt.csv", header=0, names=cols)
    else:
        df = pd.read_csv(config["dataLoc"]+config["inputFile"], header=0, names=cols)
    
    # Convert their date format to a standard date
    df["Month"] = df["Month"].apply(parser.parse)
    return df