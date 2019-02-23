from getFrequency import getFrequency

def createMmetrics(df, col):
    '''
    - Get the monthly pct change, then create a column for that change at various lags
    - Get the annual pct change    
    '''
    s = round(df[col].pct_change(periods=1),3)
    for x in range(7):
        feature = col + "_" + "M%chg_L" + str(x)
        df[feature] = s.shift(x)
    feature = col + "_" + "A%chg"
    df[feature] = round(df[col].pct_change(periods=12),3)
    return df

def createQmetrics(df, col):
    '''
    - Get the quarterly pct change, then create a column for that change at various lags
    - Get the annual pct change    
    '''
    s = round(df[col].pct_change(periods=1),3)
    for x in range(4):
        feature = col + "_" + "Q%chg_L" + str(x)
        df[feature] = s.shift(x)
    feature = col + "_" + "A%chg"
    df[feature] = round(df[col].pct_change(periods=4),3)
    return df

def genFeatures(df):
    '''
    Incoming df is all the MEIs in a DF
    Loop over the columns. For each column:
       - determine the frequency (quarterly or monthly)
       - generate additional columns holding metrics
    '''
    for col in df.columns:
        freq = getFrequency(df[col])
        if freq == "Monthly":
            df = createMmetrics(df, col)
        else:
            df = createQmetrics(df, col)
    return df