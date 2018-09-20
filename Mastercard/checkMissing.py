''' For a driver, determine if there are any missing dates.

Input is a single driver. Output is a (potentially empty) series of missing dates'''

__author__ = "The Hackett Group"

import pandas as pd

# TODO
# modify for Reported: check 3-month intervals
def checkMissing(df):
    full_range = pd.date_range(df['Month'].min(), df['Month'].max(), freq="MS")
    
    # Find the dates that aren't in the complete range
    return full_range[~full_range.isin(df['Month'])]