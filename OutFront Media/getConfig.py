''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - the number of months in the "lookback window" for identifying "wins"
    - the number of months in the "lookback window" for identifying "losses"
    - how many months to use for Test
    - an indicator allowing execution in Test mode'''

__author__ = "The Hackett Group"

def getConfig():

    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/Hackett/archive/OutFront/"
    d["fileName"]   = "3_final.csv"
    d["labelColumn"] = "population"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "OutFront.log"
    d["logDefault"] = "info"
    d["testPct"]   = 0.25
    return d