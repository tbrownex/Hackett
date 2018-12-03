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
    d["dataLoc"] = "/home/tbrownex/data/Hackett/Boeing/"
    d["originalFile"] = "MD_Base_FC_versus_Act_2015-2018_R1updated_actuals.xlsx"
    d["fileName"] = "final.csv"
    d["labelColumn"] = "Actual"
    d["logLoc"] = "/home/tbrownex/"
    d["logFile"] = "Boeing.log"
    d["logDefault"] = "info"
    d["testSize"] = 0.2
    
    d["Test"] = False
    return d