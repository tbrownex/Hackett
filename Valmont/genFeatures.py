from getFrequency import getFrequency
import numpy as np

def createMmetrics(df, col):
    '''
    - Get the monthly pct change, then create a column for that change at various lags
    - Get the annual pct change    
    '''
    #s = round(df[col].pct_change(periods=1),3).replace([np.inf, np.nan], [1.0, 0])
    s = round(df[col].pct_change(periods=1),3).replace([np.inf, np.nan], [1.0, 0])
    for x in range(4):
        feature = col + "_" + "M%chg_L" + str(x)
        df[feature] = s.shift(x)
    #feature = col + "_" + "A%chg"
    df[feature] = round(df[col].pct_change(periods=12),3)
    return df

def createWmetrics(df, col):
    '''
    - Sum by month
    - Get the weekly pct change, then create a column for that change at various lags
    '''
    monthly = df[col].resample('m').sum()
    #s = round(df[col].pct_change(periods=1),3).replace([np.inf, np.nan], [1.0, 0])
    s = round(df[col].pct_change(periods=1),3).replace([np.inf, np.nan], [1.0, 0])
    for x in range(4):
        feature = col + "_" + "M%chg_L" + str(x)
        df[feature] = s.shift(x)
    #feature = col + "_" + "A%chg"
    df[feature] = round(df[col].pct_change(periods=12),3)
    return df

def createQmetrics(df, col):
    '''
    - Get the quarterly pct change, then create a column for that change at various lags
    - Get the annual pct change    
    '''
    s = round(df[col].pct_change(periods=1),3).replace([np.inf, np.nan], [1.0, 0])
    for x in range(3):
        feature = col + "_" + "Q%chg_L" + str(x)
        df[feature] = s.shift(x)
    #feature = col + "_" + "A%chg"
    df[feature] = round(df[col].pct_change(periods=4),3)
    return df

def genFeatures(df):
    '''
    Incoming df has all the MEIs plus labels in a DF
    Loop over the columns. For each column:
       - determine the frequency (quarterly or monthly)
       - skip any columns with no frequency (like the labels)
       - generate additional columns holding metrics
    '''
    colFreq = getFrequency()
    print("\nSkipping Daily and Weekly columns for now")
    for col in df.columns:
        try:
            freq = colFreq[col]
            if freq == "M":
                df = createMmetrics(df, col)
            elif freq == "Q":
                df = createQmetrics(df, col)
            elif freq == "D":
                pass
            elif freq == "W":
                pass
        except:
            print("Skipping", col)
                 
    return df