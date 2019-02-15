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
    d["dataLoc"]     = "/home/tbrownex/data/Hackett/demos/HR/"
    d["fileName"]    = "rawData.csv"
    d["labelColumn"] = "Score"
    d["evaluationMethod"] = "--"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "demoHR.log"
    d["logDefault"] = "info"
    d["valPct"]     = 0.
    d["testPct"]    = 0.2
    d["TBdir"]      = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"]   = "/home/tbrownex/repos/Hackett/demos/HR/models"
    d["labelType"]  = "continuous"   # either "continous" or "categorical"
    return d