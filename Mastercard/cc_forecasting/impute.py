''' Fills in missing dates

For each driver:
- determine if there are missing dates. If so:
    - add the date(s)
    - set the "Amount" for each date added
    - reset the index to restore the dataframe columns as they were on input
    - sort the dataframe by UUID and Month'''

__author__ = "The Hackett Group"

import pandas as pd
import numpy  as np
from dateutil import parser
from checkMissing import checkMissing
from calcAmount   import calcAmount

# TODO
# this is for Processed only: month-offset is 1. For Reported, month-offset should be 3
# Consider what to do on gaps of more than 1 month; currently it's putting NaN

# A driver is missing dates
def addDates(uuid, driver, missing_dates):
    dfList = []           # Each Month added will be a dataframe
    d = {}
    d["UUID"]  = uuid
    
    for dt in missing_dates:
        d["Month"]  = dt
        d["Amount"] = [calcAmount(driver, dt)]
        imputed = pd.DataFrame.from_dict(d)
        imputed = imputed.set_index(["UUID"])
        dfList.append(imputed)
    return dfList

# Loop through the file, one combination at a time
def impute(df):
    for uuid, driver in df.groupby(level=0):
        missing_dates = checkMissing(driver)
        if len(missing_dates) > 0:
            dfList = addDates(uuid, driver, missing_dates)
            for x in dfList:
                df = df.append(x)
    return df

def process(df):
    df = df.set_index(["UUID"])
    df = impute(df)
    df.reset_index(inplace=True)
    df = df[["UUID", "Month", "Amount"]]
    sort = ["UUID", "Month"]
    df.sort_values(sort, inplace=True)
    return df