import pandas as pd

# input is a dataframe with a PKID stream; see if it needs imputation
def checkMissing(df):
    full_range = pd.date_range(df['Month'].min(), df['Month'].max(), freq="MS")
    # Find the dates that aren't in the complete range
    missing_dates = full_range[~full_range.isin(df['Month'])]
    
    if len(missing_dates) > 0:
        return True
    else:
        return False