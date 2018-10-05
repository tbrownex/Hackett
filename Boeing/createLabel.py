# Create the Label: the average of the 4 run times

runTimes = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]

def calcMean(df, config):
    tmp = df.loc[: , runTimes]
    
    df[config["labelColumn"]] = tmp.mean(axis=1)
    
    # Drop the individual run times
    for col in runTimes:
        del df[col]
    return df