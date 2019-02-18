''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - an indicator allowing execution in Test mode'''

__author__ = "The Hackett Group"

def getConfig():
    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/CMAPSS/"
    d["fileName"]  = "training.csv"
    d["testFile"]  = "testing.csv"
    d["labelColumn"] = "RUL"
    d["evaluationMethod"] = "--"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "demoMfg.log"
    d["logDefault"] = "info"
    d["testPct"]   = 0.     # There is a separate file with Test data
    d["TBdir"] = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/repos/Hackett/demos/Mfg/models/"  # where to save models
    return d