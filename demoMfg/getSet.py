import pandas as pd

def getSet(df, choice):
    '''
    Incoming dataframe has the data of 4 distinct tests all in one file. They can't be processed together. 
    "set" column is the identifier.
    User will have already specified which one to use in "choice"
    '''
    return df.loc[df["set"]==choice]