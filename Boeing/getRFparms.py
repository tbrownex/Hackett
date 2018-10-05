def getRFparms():
    parmList = []
    parms = {
        "n_estimators":     10,
        "max_depth":        12,
        "min_samples_split":    0.2,
        "max_features":        0.7}
    parmList.append(parms)

    parms = {
        "n_estimators":     300,
        "max_depth":        4,
        "min_samples_split":    0.2,
        "max_features":        0.7}
    parmList.append(parms)
    return parmList