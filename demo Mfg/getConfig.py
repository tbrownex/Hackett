''' 
A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - trainingFraction: during dev, train with a smaller set of data for speed
    - testPct: how many records to use for Test
'''
__author__ = "The Hackett Group"

def getConfig():

    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/CMAPSS/"
    d["fileName"]  = "trainMerged.csv"
    d["testFile"]  = "testMerged.csv"
    d["labelColumn"] = "RUL"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "MfgDemo.log"
    d["logDefault"] = "info"
    d["trainingFraction"]   = 1.0
    d["testPct"]   = 0.2
    d["nnValPct"]  = 0.2
    d["nnTestPct"] = 0.2
    d["TBdir"] = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/TF/checkpoints/"  # where to save models
    return d