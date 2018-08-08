import pandas as pd
import numpy as np
import itertools
import scipy.sparse as sp

# TODO   Issues to consider include:
# 1. consecutive missing months
# 2. recency of missing months
# 3. method of imputation
# 4. different methods for different streams

# Create a row for each missing date; set Amount = NaN
def loadNAN(df, missing_dates):
    pkid = df.index.values[0]
    for dt in missing_dates:
        tmp = {}
        tmp[pkid] = [dt, None]
        newRow = pd.DataFrame.from_dict(tmp, orient="index", columns=["Month", "Amount"])
        df = df.append(newRow)
    df = df.sort_values(["Program", "Customer", "Driver", "Month"])
    return df

# For each NAN, come up with an imputed value
# "reset_index" allows you to access the data by scalar (countable) index
# ""sp.coo_matrix" returns the index of any NaN Amounts
# The idea is to use the value of same month, previous year if you can
# Otherwise, same month following year
# Otherwise, average of previous month and following month
def calcValues(df):
    df = df.reset_index()
    _,y = sp.coo_matrix(df["Amount"].isnull()).nonzero()
    for x in y:
        if x > 11:
            val = df.iloc[x-12]["Amount"]
        else:
            if x+12 <= df.shape[0]-1:
                val = df.iloc[x+12]["Amount"]
            else:
                val = (df.iloc[x-1]["Amount"] + df.iloc[x+1]["Amount"])/2
        df.iloc[x,4] = val
    return df
    
# input is a dataframe with a PKID stream; see if it needs imputation
def impute(df):
    full_range = pd.date_range(df['Month'].min(), df['Month'].max(), freq="MS")
    # Find the dates that aren't in the complete range
    missing_dates = full_range[~full_range.isin(df['Month'])]
    if len(missing_dates) > 2:            # Put a limit on how many records to impute
        return False, df
    else:
        df = loadNAN(df, missing_dates)
        df = calcValues(df)
        return True, df