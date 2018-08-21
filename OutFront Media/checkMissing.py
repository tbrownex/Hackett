import pandas as pd

# input is a dataframe for a panel/date; see if it needs imputation
# "day" will check if any days are missing
# "hour" checks for any missing times
def checkMissing(df, typ):
    assert (typ in ["day", "hour"]), "invalid typ passed to checkMissing"
    if typ == "day":
        full_range = pd.date_range(df['date'].min(), df['date'].max(), freq="D")
        missing = full_range[~full_range.isin(df['date'])]
    else:
        full_range = pd.Series([x for x in range(24)])
        missing = full_range[~full_range.isin(df['hour'])]
    
    return missing