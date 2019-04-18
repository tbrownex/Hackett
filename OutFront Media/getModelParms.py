import itertools
''' This is the set of parameters we're "gridsearching" when optimizing the algos '''

def getParms(typ):
    if typ == "RF":
        nEstimators      = [50]
        min_samples_split = [25]
        max_depth         = [10]
        min_samples_leaf  = [4]
        max_features      = [ 0.9]
        return list(itertools.product(nEstimators,
                                      min_samples_split,
                                      max_depth,
                                      min_samples_leaf,
                                      max_features))
    elif typ == "XGB":
        colsample_bytree  = [0.6]
        learningRate      = [5e-3]
        maxDepth          = [10]
        nEstimators       = [50, 100]
        alpha             = [0.3]
        return list(itertools.product(colsample_bytree,
                                      learningRate,
                                      maxDepth,
                                      nEstimators,
                                      alpha))
    elif typ == "STL":
        window = ["periodic"]
        return list(itertools.product(window))
    elif typ == "NN":
        L1Size       = [12]
        activation   = ["tanh"]
        batchSize    = [64]
        learningRate = [1e-3]
        std          = [0.25]
        dropout      = [0.3]
        optimizer    = ["Adam"]
        return list(itertools.product(L1Size,
                                      activation,
                                      batchSize,
                                      learningRate,
                                      std,
                                      dropout,
                                      optimizer))
    elif typ == "AE":     # AutoEncoder for outlier detection
        L1Size       = [12,16]
        L2Size       = [12,14]
        activation   = ["relu"]
        batchSize    = [32]
        learningRate = [1e-3]
        std          = [0.25]
        dropout      = [0.45]
        optimizer    = ["Adam"]
        return list(itertools.product(L1Size,
                                      L2Size,
                                      activation,
                                      batchSize,
                                      learningRate,
                                      std,
                                      dropout,
                                      optimizer))