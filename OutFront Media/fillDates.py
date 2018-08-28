from datetime import timedelta
from checkMissing import checkMissing

# Use the population and hours from the week prior, or following,
# for this record
def getVals(df, d):    
    try:
        dt = d["date"] - timedelta(days=7)
        tmp = df.loc[d["panel"], dt]
    except:
        dt  = d["date"] + timedelta(days=7)        
        tmp = df.loc[d["panel"], dt]
    pop   = tmp["population"].reset_index(drop=True)
    hours = tmp["hour"].reset_index(drop=True)
    return pop, hours

# A panel is missing dates
def appendDates(panel, df, missing):
    d = {}
    d["panel"] = panel
    
    for dt in missing:
        d["date"]  = dt
        d["population"], d["hour"] = getVals(df, d)
        imputed = pd.DataFrame.from_dict(d)
        imputed = imputed.set_index(["panel", "date"])
        dfList.append(imputed)
    return dfList

def fillDates(panel, df):
    print(panel)
    missing = checkMissing(df, "day")
    df.set_index(["panel", "date"], inplace=True)
    dfList = appendDates(panel, df, missing)
    for x in dfList:
        df = df.append(x)
    return df