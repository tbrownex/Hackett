import pandas as pd

# pkid: Country, Program, Parent ICA, OrigMeasure
# typ:  Processed or Reported
def getDriver(pkid, typ):
    if typ == "P":
        df = pd.read_csv("c:\\Users\\tbrowne\\Python\\Processed.csv")
    else:
        df = pd.read_csv("c:\\Users\\tbrowne\\Python\\Reported.csv")
    df = df.set_index(["Country", "Program", "Parent ICA", "OrigMeasure"])
    
    key = pkid[0]+","+pkid[1]+","+pkid[2]+","+pkid[3]
    print(pkid)
    tmp = df.loc[pkid][["Month", "Value"]]
    tmp = tmp.reset_index(drop=True)
    
    dates  = tmp["Month"]
    values = tmp["Value"]
    return (dates, values)