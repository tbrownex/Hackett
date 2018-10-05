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