''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - how many months to use for Test'''

__author__ = "The Hackett Group"

def getConfig():

    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/Hackett/Valmont/"
    d["fileName"]   = "ValmontData.csv"
    d["labelColumn"] = "Volume"
    d["labelType"]  = "continuous"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "Valmont.log"
    d["logDefault"] = "info"
    d["valPct"]     = 0.
    d["testMonths"] = 12
    d["lagMonths"] = 3
    # --- begin ETS parameters ---
    d["numForecasts"]= 6
    # --- end of ETS ---
    return d