import pandas as pd
import numpy  as np
import time
from config import getClient
import time

numRecords = 1000000
keep       = ['panelid', 'event_date_dt', 'hourinterval', 'populationcount']

def getLoc():
    return getClient("OutFront")

def processChunk(df, dataloc):
    # Rename the columns
    cols = ["panel", "date", "hour", "population"]
    df.columns = cols
    
    df.set_index("panel", inplace=True)
    grp = df.groupby(level=0)
    for key, panel in grp:
        panel.to_csv(dataloc+"panels/"+key+".csv")

def splitPanels():
    dataloc = getLoc()
    start = time.time()
    count = 0

    for chunk in pd.read_csv(dataloc+"OutFront1000_sorted.csv",\
                             usecols=keep, dtype={"hourinterval":'int8',\
                                                  "populationcount":'int32'},\
                             chunksize=numRecords):
        count += 1
        processChunk(chunk, dataloc)
    
    end = time.time()  
    print("Finished after {:.0f} minutes".format((end-start)/60))
    
if __name__== "__main__":
    splitPanels()