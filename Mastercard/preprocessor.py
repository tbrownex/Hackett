''' Prepare the data for modeling:
- remove any drivers not in-scope
- concatenate Country, Program, Customer and Driver to create single UUID field
- generate missing dates'''

__author__ = "The Hackett Group"

import pandas as pd
import logging
import censor
import impute
import createUUID
import formatColumns
import normalize

def process(df, config, args):
    df = censor.process(df, config)
    df = createUUID.process(df)
    df = formatColumns.process(df)
    df = impute.process(df)
    if args.normalize == "zadj":
        logging.info("Normalizing input")
        df = normalize.process(df)
    return df