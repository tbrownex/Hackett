def getXGBparms():
    parmList = []
    parms = {
        "n_estimators":     30,
        "max_depth":        12,
        "learning_rate":    0.2,
        "subsample":        0.7,
        "colsample_bytree": 1.0}
    parmList.append(parms)

    parms = {
        "n_estimators":     300,
        "max_depth":        12,
        "learning_rate":    0.2,
        "subsample":        0.7,
        "colsample_bytree": 1.0}
    parmList.append(parms)
    return parmList