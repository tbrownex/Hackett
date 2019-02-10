import itertools
from tensorflow import keras

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
        nEstimators      = [100]
        learningRate     = [1e-2, 1e-3]
        maxDepth         = [6]
        min_child_weight = [10]
        colsample_bytree = [0.8]
        subsampling      = [0.8]
        gamma            = [0]
        return list(itertools.product(nEstimators,
                                      learningRate,
                                      maxDepth,
                                      min_child_weight,
                                      colsample_bytree,
                                      subsampling,
                                      gamma))
    elif typ == "NN":
        L1Size       = [12]
        activation   = ["tanh"]
        batchSize    = [64]
        learningRate = [1e-3]
        std          = [0.25]
        dropout      = [0.4]
        optimizer    = ["Adam"]
    return list(itertools.product(L1Size,
                                  activation,
                                  batchSize,
                                  learningRate,
                                  std,
                                  dropout,
                                  optimizer))