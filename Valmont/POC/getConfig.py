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
    d["dataLoc"]     = "/home/tbrownex/data/Hackett/Valmont/POC/"
    d["fileName"]    = "testData.csv"
    d["labelColumn"] = "Label"
    d["labelType"]   = "continuous"
    d["logLoc"]      = "/home/tbrownex/"
    d["logFile"]     = "Valmont.log"
    d["logDefault"]  = "info"
    d["valPct"]      = 0.15
    d["testPct"]     = 0.25
    d["TBdir"]       = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"]    = "/home/tbrownex/repos/Hackett/Valmont/POC/models/"  # where to save models
    return d