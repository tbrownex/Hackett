''' "Win" is when you have a recent customer so not enough data to forecast
"Loss" is when you don't have recent data, so no longer active/relevant

For each Win/Loss, report on its removal from the data'''

__author__ = "The Hackett Group"

import pandas as pd
import datetime
import dateutil.relativedelta
import reportWriter

def fmtRec(idx, typ):
    key = "|".join(idx)
    return key+"|"+typ
    
def process(df, config):
    today = datetime.datetime.today()
    lossMonth = today - dateutil.relativedelta.relativedelta(months=config["lossMonths"])
    winMonth  = today - dateutil.relativedelta.relativedelta(months=config["winMonths"])
    
    sort = ["Country", "Program", "Customer", "Driver", "Month"]
    df = df.sort_values(sort)
    df = df.set_index(["Country", "Program", "Customer", "Driver"], drop=True)
    
    rpt = reportWriter.Report(config["dataLoc"], "WinsLosses.csv")
    
    hdr = ["Country", "Program", "Customer", "Driver", "Type"]
    hdr = "|".join(hdr)
    rpt.writeRow([hdr])    
    
    # Process a combination at a time
    for idx, val in df.groupby(level=[0, 1, 2, 3]):
        dates = val["Month"]
        dateutil.relativedelta.relativedelta(months=config["lossMonths"])
        if dates.iloc[-1] < lossMonth:    # Check for Loss
            rec = fmtRec(idx, "L")
            rpt.writeRow([rec])
            df.drop(idx, inplace=True)
        else:                        # Check for Win
            if dates.iloc[0] > winMonth:
                rec = fmtRec(idx, "W")
                rpt.writeRow([rec])
                df.drop(idx, inplace=True)
    rpt.close()
    return df.reset_index()