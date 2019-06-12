''' We have just generated a Month for a driver. Calculate the Amount by taking
    the average of the previous and following months'''

__author__ = "The Hackett Group"

from dateutil.relativedelta import *

# TODO
# Figure out what to do with error handling (no previous or following month)

# Use the average of the months prior and following
def calcAmount(df, dt):
    df.reset_index(inplace=True)
    df.set_index("Month", inplace=True)
    found = True
    
    foll = dt + relativedelta(months=  1)
    prev = dt + relativedelta(months= -1)
    
    try:
        p = df.loc[prev]["Amount"]
        f = df.loc[foll]["Amount"]
    except:
        #error handler
        found = False
            
    if found:
        return (p+f)/2
    else:
        return None