import pandas as pd

def process(df, config):
    train = df.head(-config["testMonths"])
    test  = df.head( config["testMonths"])
    return train, test