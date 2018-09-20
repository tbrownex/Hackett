''' This is the point of entry for forecasating of revenues.
    Input file will consist of Country/Program/Customer/Driver combinations.
    Combinations are processed in isolation, i.e. no mixing of combinations occurs.
    
    This module will:
    - get the command line arguments
    - get the Config file
    - set the Logging level
    - read the input file into a dataframe. This dataframe is then passed around, \
    modified and eventually persisted
    - kick off the processing of the input file'''
    
__author__ = "The Hackett Group"

from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
from setLogging import setLogging
import logging
import preprocessor
import train

def process(df, config, args):
    df = preprocessor.process(df, config, args)
    if args.trainInd == "train":
        df = train.process(df, config, args)
    df = forecast.process(df)
    return df

if __name__ == "__main__":    
    args   = getArgs()
    config = getConfig()
    setLogging(config, args)
    logging.info("Start of a run")
    
    df = getData(config)
    df = process(df, config, args)
    df.to_csv(config["dataLoc"]+"Processed.csv", sep="|", index=False)
    
    logging.info("End of a run")