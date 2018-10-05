''' Calculate the MAPE of predictions vs test.

"predictions" is a numpy array
"test" is a series'''

import numpy as np

def process(predictions, dataDict):
    test = dataDict["testY"]
    ensemble = np.mean(predictions, axis=1)
    return np.mean(np.abs((ensemble - test) / test))