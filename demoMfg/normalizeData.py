from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize(dataDict, method):
    '''
    Normalize the input, which should mean the sensor readings, cycle and any features generated from
    the sensor readings.
    Do not normalize the RUL or Unit
    '''
    assert method in ['MinMaxA','MinMaxB', 'Std', 'tanh'],  "Invalid scaling method"
    
    if method =='MinMaxA':
        min_max_scaler = MinMaxScaler()
        data           = pd.DataFrame(min_max_scaler.fit_transform(dataDict))
        data.columns   = sv_cols
        scale_range    = min_max_scaler.data_range_[-1]
        scale_min      = min_max_scaler.data_min_[-1]
        return data, scale_range, scale_min
    elif method =='MinMaxB':
        arr        = dataDict.values
        width      = np.ptp(arr, axis=0)
        scale_min  = np.min(arr, axis=0)
        norm       = (2 * (arr - scale_min) / width) -1
        data       = pd.DataFrame(norm, columns=sv_cols)
        return data, width[-1], scale_min[-1]
    elif method == 'Std':
        # StandardScaler returns a numpy array so need to recreate the DF
        cols = dataDict["trainX"].columns
        npArray            = StandardScaler().fit_transform(dataDict["trainX"])
        dataDict["trainX"] = pd.DataFrame(npArray, columns=cols)
        # Test has the Unit as a column which needs to be removed (not normalized) then added back
        svUnits = dataDict["testX"]["unit"]
        del dataDict["testX"]["unit"]
        npArray            = StandardScaler().fit_transform(dataDict["testX"])
        dataDict["testX"]  = pd.DataFrame(npArray, columns=cols)
        dataDict["testX"]["unit"] = svUnits
        return dataDict
    else:
        arr = dataDict.values
        std = np.std(arr,axis=0)
        avg = np.mean(arr,axis=0)
        Z = (arr - avg) / std
        norm = 0.5 * (np.tanh( Z * .01 ) + 1)
        data = pd.DataFrame(norm, columns=sv_cols)
        return data, std[-1], avg[-1]