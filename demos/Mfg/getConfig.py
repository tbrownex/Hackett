''' A dictionary object that holds key parameters such as:
    - the location and name of the input data file
    - the location of the JSON files (interface with demo UI)
    - the name of the Label column
    - the location and name of the log file
    - the default logging level
    - the location of Tensorboard data
    - the location where models should be stored '''

__author__ = "The Hackett Group"

def getConfig():
    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/Hackett/demos/CMAPSS/"
    d["JSONloc"] = "/home/tbrownex/repos/Hackett/demos/Mfg/"
    d["fileName"]  = "train.csv"
    d["testFile"]  = "test.csv"
    d["labelColumn"] = "RUL"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "demoMfg.log"
    d["logDefault"] = "info"
    d["TBdir"] = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/repos/Hackett/demos/Mfg/models/"  # where to save models
    return d