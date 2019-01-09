import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from genFeatures import genFeatures

def partition(df, config):
    assert config["nnTestPct"] > 0 or config["nnValPct"] > 0, "must specify either a test pct or validation pct"
    sv_cols = df.columns
    
    train, test = train_test_split(df, test_size=config["nnTestPct"])
    test.columns = sv_cols
    if config["nnValPct"] > 0:
        train, val = train_test_split(train, test_size=config["nnValPct"])
        train.columns = sv_cols
        val.columns   = sv_cols
        return train, val, test
    else:
        train.columns = sv_cols
        return train, None, test

def splitLabels(df, config):
    ''' Separate the features and labels  '''
    labels = np.array(df[config["labelColumn"]])
    labels = np.reshape(labels, newshape=(-1,1))
    del df[config["labelColumn"]]
    return df, labels

def normalize(data, method):
    '''
    Normalize the input, which should mean the sensor readings, cycle and any features generated from
    the sensor readings.
    Do not normalize the RUL or Unit
    '''
    assert isinstance(data, pd.DataFrame),   "data must be pandas DataFrame"
    assert method in ['MinMaxA','MinMaxB', 'Std', 'tanh'],  "Invalid scaling method"
    skip = ["RUL", "unit"]
    norm = [col for col in data.columns if col not in skip]
    
    if method =='MinMaxA':
        min_max_scaler = MinMaxScaler()
        data           = pd.DataFrame(min_max_scaler.fit_transform(data))
        data.columns   = sv_cols
        scale_range    = min_max_scaler.data_range_[-1]
        scale_min      = min_max_scaler.data_min_[-1]
        return data, scale_range, scale_min
    elif method =='MinMaxB':
        arr        = data.values
        width      = np.ptp(arr, axis=0)
        scale_min  = np.min(arr, axis=0)
        norm       = (2 * (arr - scale_min) / width) -1
        data       = pd.DataFrame(norm, columns=sv_cols)
        return data, width[-1], scale_min[-1]
    elif method == 'Std':
        scaled_features = data.copy()
        toBeScaled = scaled_features[norm]
        scaled = StandardScaler().fit_transform(toBeScaled.values)
        scaled_features[norm] = scaled
        return scaled_features
    else:
        arr = data.values
        std = np.std(arr,axis=0)
        avg = np.mean(arr,axis=0)
        Z = (arr - avg) / std
        norm = 0.5 * (np.tanh( Z * .01 ) + 1)
        data = pd.DataFrame(norm, columns=sv_cols)
        return data, std[-1], avg[-1]
    
def selectCols(df):
    '''
    Ignore the generated features done for Random Forest
    Ignore "settings" because too much variation across the different units/engines
    '''
    keep = ['unit','cycle','setting1','setting2','setting3','sensor1','sensor2','sensor3',\
            'sensor4','sensor5','sensor6',\
            'sensor7','sensor8','sensor9', 'sensor10', 'sensor11', 'sensor12',\
            'sensor13','sensor14', 'sensor15','sensor16', 'sensor17', 'sensor18', 'sensor19',\
            'sensor20','sensor21','RUL']
    return df[keep]
    
def prepData(df, config, args):
    df = selectCols(df)
    if args.genFeature == "Y":
        df = genFeatures(df)
    df = df.dropna()           # First 4 rows of each Unit will be "NA" due to rolling calculation
    df = normalize(df, "Std")
    df = df.sample(frac=config["trainingFraction"])
    del df["unit"]
    train, val, test = partition(df, config)
    
    dataDict = {}
    features, labels = splitLabels(train, config)
    dataDict["trainX"] = features
    dataDict["trainY"] = labels
    features, labels = splitLabels(test, config)
    dataDict["testX"] = features
    dataDict["testY"] = labels
    if config["nnValPct"] > 0:
        features, labels = splitLabels(val, config)
        dataDict["valX"] = features
        dataDict["valY"] = labels
    return dataDict