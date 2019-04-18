''' A dictionary object that holds key parameters such as:
    - the location and name of the input data file
    - the name of the Label column
    - the location and name of the log file
    - the default logging level
    - the location of Tensorboard data
    - the location where models should be stored '''

__author__ = "The Hackett Group"

def getConfig():
    d = {}
    d["dataLoc"]     = "/home/tbrownex/data/LV/"
    d["fileName"]    = "HH+neighborhood_rank.csv"
    d["labelColumn"] = "Label"
    d["labelType"]   = "categorical"
    d["numClasses"]  = 2
    d["logLoc"]      = "/home/tbrownex/"
    d["logFile"]     = "demoNN.log"
    d["logDefault"]  = "info"
    d["TBdir"] = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/repos/Hackett/demos/NN/models/"  # where to save models
    d["valPct"]    = 0.15
    d["testPct"]   = 0.25
    return d