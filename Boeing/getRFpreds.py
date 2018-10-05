from sklearn.ensemble import RandomForestRegressor

def process(parms, dataDict):
    regr = RandomForestRegressor(**parms)
    model = regr.fit(dataDict["trainX"], dataDict["trainY"])
    return model.predict(dataDict["testX"])