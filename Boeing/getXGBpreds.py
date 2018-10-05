''' Create an XGBoost model on training data and return predictions from test data

"parms" is a dictionary
"dataDict" is a dictionary where:
   - X are dataframes
   - Y are Series '''

import xgboost as xgb

def process(parms, dataDict):    
    model = xgb.XGBRegressor(**parms)
    model = model.fit(dataDict["trainX"], dataDict["trainY"])
    return model.predict(dataDict["testX"])