''' Replace several fields with a concatenated, single field to serve as the "key" field.

Columns concatenated are: Country, Program, Customer, Driver
Delete the fields used to generate UUID'''

__author__ = "The Hackett Group"

import pandas as pd

def process(df):
    cols = ["Country", "Program", "Customer", "Driver"]
    df["UUID"] = df[cols].apply(lambda x: '|'.join(x), axis=1)
    return df