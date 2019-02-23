import itertools

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
    elif typ == "NN":
        L1Size       = [12, 48]
        Lambda       = [0, 0.1]
        activation   = ["tanh", "ReLU"]     # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'
        batchSize    = [16,64]
        learningRate = [5e-4, 1e-4]
        std          = [0.05, 0.2]
        return list(itertools.product(L1Size,
                                      Lambda,
                                      activation,
                                      batchSize,
                                      learningRate,
                                      std))