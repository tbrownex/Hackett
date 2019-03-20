from getConfig import getConfig
import time 

def getData(config):
    df = pd.read_csv(config["demoLoc"]+"predDF.csv")
    df.drop(columns=["Baseline", "actual"], inplace=True)
    df.set_index("unit", inplace=True)
    # These are the units we want to demo
    keep = [2,3,4,34]
    df = df.loc[keep]
    
    df["RUL"] = df.mean(axis=1)
    
    df.drop(columns=["RF", "NN", "XGB"], inplace=True)
    return df

def getStreams(df, start, units):
    L = []
    for unit in units:
        s = df.loc[unit][:-start]["RUL"]
        s.reset_index(drop=True, inplace=True)
        L.append(s)
    merged = pd.concat(L, ignore_index=True,axis=1)
    merged.columns = units
    return merged

if __name__ == "__main__":
    df  = getData()
    print(df.shape)
    input()
    config = getConfig()
    print(config)
    input()
    # Get the shortest stream: start the loop at this point
    unitCounts = df.groupby("unit").count()
    start = unitCounts.min()[0]
    units = np.unique(df.index.values)
    print("start: ", start)
    print("units: ", units)
    input()
    while start > 0:
        latest = getStreams(df, start, units)
        latest.to_csv(config["demoLoc"] + "MLdemo data.csv", index=False)
        time.sleep(5)
        start -=1