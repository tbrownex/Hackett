import pandas as pd
import numpy  as np
import time
from config    import getClient
from getPanels import getPanels
import time
import os
import fillDates as fd

dataloc = getClient("OutFront")
errors  = open(dataloc+"errors.csv", "w")

def getDuration(df):
    start = df["date"].min()
    end   = df["date"].max()
    return (end - start)/ np.timedelta64(1, 'M')

def getFillRate(df):
    dates = df["date"]
    diff  = (dates.max() - dates.min()).days
    durations = diff*24
    numPoints = len(dates)
    return numPoints/durations

def processError(panel, df, msg, dataloc):
    os.rename(dataloc+"panels/"+panel,\
              dataloc+"errors/"+panel)

    rec = panel+","+msg+"\n"
    errors.write(rec)

def dataPrep(panel, df, dataloc):
    df = fd.fillDates(panel, df)
    
def processPanel(panel, df, dataloc):
    # Convert their date format to a standard date
    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d')
    months = getDuration(df)
    if months > 13:
        rate = getFillRate(df)
        if rate > .90:
            dataPrep(panel, df, dataloc)
        else:
            processError(panel, df, "Fill Rate too low", dataloc)
    else:
        processError(panel, df, "Duration too short", dataloc)

def processPanels(panels, dataloc):
    
    count = 0
    start = time.time()

    for panel in panels:
        count += 1
        if count %100 == 0: print(count)
        df = pd.read_csv(dataloc+"panels/"+panel,\
                         dtype={"hour":'int8',\
                                "population":'int32'})
        processPanel(panel, df, dataloc)
        
    errors.close()
    end = time.time()
    print("Completed after {:.0f} minutes".format((end-start)/60))

if __name__== "__main__":
    panels  = getPanels(dataloc)
    processPanels(panels, dataloc)