import itertools

def getParms(typ):
    if typ == "RF":
        n_estimators      = [10]
        min_samples_split = [10]
        max_depth         = [10]
        min_samples_leaf  = [4]
        max_features      = [0.5]
        return list(itertools.product(n_estimators,
                                      min_samples_split,
                                      max_depth,
                                      min_samples_leaf,
                                      max_features))
    elif typ == "NN":
        ''' NN hyper-parameters '''
        l1Size       = [16,32,64,128]           # Count of nodes in layer 1
        #l2_size       = [64, 128, 256]          # Count of nodes in layer 2
        learningRate = [0.005]
        Lambda        = [0, 0.04]          # Regularization parameter
        batchSize    = [32]
        epochs        = [20]
        activation    = ['ReLU']           # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'
        stdDev    = [0.1]            # StdDev for initializing weights
    
        for x in activation:
            assert x in ['tanh', 'leakyReLU', 'ReLU', 'ReLU6'], "Invalid Activation: %s" % x
        
            return list(itertools.product(l1Size,
                                          learningRate,
                                          Lambda,
                                          batchSize,
                                          epochs,
                                          activation,
                                          stdDev))