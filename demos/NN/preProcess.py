''' Prepare the data for modeling:
    - Shuffle the Training data
    - Remove non-features "set" and "unit"
    - Identify any static columns (single value) and remove them
    - Split the features from the labels
    - Normalize the data
    - (optional) Remove outliers
    - No need to train/test split: there's a separate file for Test
    '''
__author__ = "Tom Browne"

import pandas as pd
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from analyzeCols   import analyzeCols
from splitData     import splitData
from splitLabel    import splitLabel
#from removeOutliers import removeOutliers

def removeCols(df):
    ''' if any column is constant (same value for all rows) remove it '''
    cols   = df.columns
    remove = analyzeCols(df)
    keep = [col for col in cols if col not in remove]
    df = df[keep]
    return df

def convertCategoricals(df):
    cols = ["marital_status","owner_type"]
    return pd.get_dummies(df, columns=cols)

def standardize(df):
    ''' Z-score columns, ignoring the binary columns '''
    cols = ["age_alt","income_alt","home_value","num_hh","num_adults",\
            "num_kids","net_worth_alt","density","age_bins","income_bins",\
            "hv_bins","density_bins","neighborhood_bin"]
    dfCopy   = df.copy()
    features = dfCopy[cols]
    scaler   = StandardScaler()
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    dfCopy[cols] = features
    return dfCopy
        
#def preProcess(train, test, config, args):
def preProcess(df, config):
    '''
    - Remove constant columns (in case we missed any from prev step)
    - Remove rows with blanks (NaN)
    - Convert categorical columns to "one-hot"
    - Standardize
    - Create Train/Val/Test partitions
    - Split features and labels
    - Normalize
    - (optional) remove outliers
    '''
    df = removeCols(df)
    df = df.dropna(axis=0, how='any')
    df = convertCategoricals(df)
    df = standardize(df)
    dataDict = splitData(df, config)
    dataDict = splitLabel(dataDict, config)
    '''if args.Outliers == "Y":
        dataDict = removeOutliers(dataDict, config)'''
    return dataDict