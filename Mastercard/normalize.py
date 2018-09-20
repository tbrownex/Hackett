''' For each UUID, normalize (Z-score) the Amount column.
ddof is set to 1 in the std function. '''

__author__ = "The Hackett Group"

import pandas as pd

def process(df):
    df["Amount"] = (df["Amount"] - df["Amount"].mean())/df["Amount"].std(ddof=1)
    return df