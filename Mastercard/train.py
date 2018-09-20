''' For each UUID:
- split data into Train and Test
'''

__author__ = "The Hackett Group"

import pandas as pd
import split
import model
import forecast
import evaluate

def process(df, config, args):
    df = df.set_index(["UUID"])
    for uuid, driver in df.groupby(level=0):
        train, test = split.process(driver, config)
        m = model.process(train, args)
        predictions = forecast.process(m, config, args)
        MAPE        = evaluate.process(predictions, test)
        print(MAPE)