''' For a simplistic baseline prediction use the Avg Time to Fail for the Training data:
    - 20,600 rows of Training data
    - 100 Units
    - Avg = 206 cycles to failure
    
    So in the Test data, after 100 cycles, you would say RUL = 206-100=106
    '''
import pandas as pd
import numpy  as np

def getBaselinePreds(dataDict):
    mean = 206   # I used Excel: 20,600 rows for 100 units
    baseLine = pd.Series([x for x in range(mean, -200, -1)])
    
    test = dataDict["testX"].copy()
    test.set_index("unit", inplace=True)
    grp = test.groupby(level=0)
    
    dfList = []
    for k,v in grp:
        length = v.shape[0]
        v["baseline"] = baseLine.values[:length]
        v.reset_index(inplace=True)
        dfList.append(v)
    df = pd.concat(dfList)
    preds = df["baseline"]
    preds = np.array(preds)
    return np.reshape(preds, newshape=[-1,1])