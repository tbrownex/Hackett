''' Filter the dataset, keeping only the countries specified.

Input is the complete set of drivers. Output is a subset, after excluding certain countries.'''

__author__ = "The Hackett Group"

import pandas as pd

countries = ['United Arab Emirates', 'South Africa', 'Saudi Arabia',\
             'Nigeria', 'Egypt','Qatar', 'Lebanon', 'Kuwait',\
             'Pakistan', 'Jordan','Libya', 'Tunisia']

# Select only the top 12 countries
def process(df):
    return df.loc[df["Country"].isin(countries)]