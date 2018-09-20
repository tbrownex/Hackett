''' Select only the columns we want for the rest of the process. 

Incoming dataframe has Country, Program, Customer and Driver:
- Drop the Country, Program, Customer and Driver columns
- Reorder to: UUID, Month, Amount'''

__author__ = "The Hackett Group"

import pandas as pd

cols = ["UUID", "Month", "Amount"]

def process(df):
    return df[cols]