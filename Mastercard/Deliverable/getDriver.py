import pandas as pd
from config import getClient

# pkid: Country, Program, Parent ICA, OrigMeasure
# typ:  Processed or Reported
def getDriver(pkid, typ):
    dataloc = getClient("MC")
    country = pkid[0]
    program = pkid[1]
    ica     = pkid[2]
    driver  = pkid[3]
    if typ == "P":
        #df = pd.read_csv(dataloc+"ref/"+"CleanDriverFile.csv")
        df = pd.read_csv(dataloc+"Processed.csv")
    else:
        df = pd.read_csv(dataloc+"Reported.csv")
    df = df.set_index(["Country", "Program", "Customer", "Driver"])
    
    key = ', '.join('"{0}"'.format(parm) for parm in pkid)
    tmp = df.loc[country, program, ica, driver][["Month", "Amount"]]
    tmp = tmp.reset_index(drop=True)
    
    dates  = tmp["Month"]
    values = tmp["Amount"]
    return (dates, values)