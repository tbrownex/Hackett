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
    d["dataLoc"]    = "/home/tbrownex/data/Hackett/Mastercard/"
    d["inputFile"]  = "mea-processed_1Aug2018.csv"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "MC.log"
    d["logDefault"] = "info"
    d["lossMonths"] = 7
    d["winMonths"]  = 12
    d["censorFile"] = "WinsLosses.csv"
    d["testMonths"] = 6
    d["Test"]       = True
    return d