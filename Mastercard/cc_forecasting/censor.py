''' Removes drivers that should not be forecast. For instance, we remove "Wins" and "Losses" due to insufficient data (wins) or not relevant (losses).
We also can remove entire countries, if that is required.

Input is the complete set of drivers. Output is a subset, after censoring.'''

__author__ = "The Hackett Group"

import filterCountries
import removeWinsandLosses
# This module is out of date

def process(df, config):
    df = filterCountries.process(df)
    df = removeWinsandLosses.process(df, config)
    return df