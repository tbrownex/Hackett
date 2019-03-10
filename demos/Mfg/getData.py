import pandas as pd

__author__ = "Tom Browne"

def getData(config, sep=","):
    train = pd.read_csv(config["dataLoc"]+config["fileName"], sep=sep)
    test  = pd.read_csv(config["dataLoc"]+config["testFile"], sep=sep)
    return train, test