import pandas as pd

def meanDiff(df):
    '''
    For each sensor, get the mean from cycles that have >100 RUL. Then for each reading, compute the
    difference between the current reading and that mean.
    '''
    cols = ["sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",\
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16",\
            "sensor17","sensor18","sensor19","sensor20","sensor21"]
    dfList = []
    df.set_index("unit", inplace=True)
    grp = df.groupby(level=0)
    
    for key, val in grp:
        for c in cols:
            mean = val[c][:10].mean()
            newCol = "mean_"+c
            val = val.assign(tmp = val[c]-mean)
            val.rename(index=str, columns={"tmp": newCol}, inplace=True)
        val.reset_index(inplace=True)
        dfList.append(val)
    df = pd.concat(dfList, ignore_index=True)
    return df

def movingMean(df):
    '''
    For each sensor, get the mean from cycles that have >100 RUL. Then for each reading, compute the
    difference between the current reading and that mean.
    '''
    cols = ["sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",\
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16",\
            "sensor17","sensor18","sensor19","sensor20","sensor21"]
    dfList = []
    df.set_index("unit", inplace=True)
    grp = df.groupby(level=0)
    
    for key, val in grp:
        for c in cols:
            newCol = "movingMean_"+c
            movingMean = val[c].rolling(window=5).mean()
            val = val.assign(tmp = movingMean)
            val.rename(index=str, columns={"tmp": newCol}, inplace=True)
        val.reset_index(inplace=True)
        dfList.append(val)
    df = pd.concat(dfList, ignore_index=True)
    return df

def movingStd(df):
    '''
    For each sensor, get the rolling std deviation from
    '''
    cols = ["sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",\
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16",\
            "sensor17","sensor18","sensor19","sensor20","sensor21"]
    dfList = []
    df.set_index("unit", inplace=True)
    grp = df.groupby(level=0)
    
    for key, val in grp:
        for c in cols:
            newCol = "movingStd_"+c
            movingStd = val[c].rolling(window=10).std()
            val = val.assign(tmp = movingStd)
            val.rename(index=str, columns={"tmp": newCol}, inplace=True)
        val.reset_index(inplace=True)
        dfList.append(val)
    df = pd.concat(dfList, ignore_index=True)
    return df

def genFeatures(df):
    #df = meanDiff(df)
    #df = movingMean(df)
    df = movingStd(df)
    return df