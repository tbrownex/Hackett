import h2o
import pandas as pd
from sklearn.model_selection import train_test_split

# These fields are useless in predicting:
# "Job", "Year", "Work Site", "BD Status", "Major EAC", "Income Statement Type", 
# "Calc Type", "Revenue Calc Type", "Auto Earn Adj", "Award Fee" 
# The list holds the index values of these columns
def dropCols(df):
    df = df.drop([0, 1, 11, 16, 20, 21, 22, 23, 24, 26])
    return df

def splitYears(df):
    ''' 
    We'll use 2016 for training and 2017 for Test because that's how this would work in production
    '''
    mask = df["Year"] == 2016
    train = df[mask, :]
    
    mask = df["Year"] == 2017
    test = df[mask, :]
    return train, test

def splitTraining(df):
    ''' This is not the same as sklearn splitting
    H2O will split one more time than the number of partition sizes given:
    https://www.rdocumentation.org/packages/h2o/versions/3.20.0.8/topics/h2o.splitFrame
    Hard-coding these values because this is just exploratory
    '''
    train, val = df.split_frame([0.8])
    d = {}
    d["train"] = train
    d["val"]   = val
    return d

def prepData(df, config):
    train, test = splitYears(df)
    train = dropCols(train)
    test  = dropCols(test)
    d = splitTraining(train)
    d["test"]  = test
    return d