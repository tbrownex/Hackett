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
        nEstimators       = [100]
        learningRate      = [2e-2]
        maxDepth          = [5]
        min_child_weight  = [8]
        colsample_bytree  = [0.55]
        subsampling       = [1.0]
        gamma             = [0]
        return list(itertools.product(nEstimators,
                                      learningRate,
                                      maxDepth,
                                      min_child_weight,
                                      colsample_bytree,
                                      subsampling,
                                      gamma))
    elif typ == "NN":
        L1Size       = [12]
        activation   = ["ReLU"]
        Lambda       = [1e-2]
        batchSize    = [64]
        learningRate = [1e-4]
        std          = [0.25]
        dropout      = [0.3]
        optimizer    = ["Adam"]
        weight       = [1]
        return list(itertools.product(L1Size,
                                      activation,
                                      Lambda,
                                      batchSize,
                                      learningRate,
                                      std,
                                      dropout,
                                      optimizer,
                                      weight))
    elif typ == "AE":     # AutoEncoder for outlier detection
        L1Size       = [18]
        L2Size       = [12]
        activation   = ["tanh"]
        batchSize    = [32]
        learningRate = [2e-3]
        std          = [0.5]
        dropout      = [0.4]
        optimizer    = ["Adam"]
        return list(itertools.product(L1Size,
                                      L2Size,
                                      activation,
                                      batchSize,
                                      learningRate,
                                      std,
                                      dropout,
                                      optimizer))