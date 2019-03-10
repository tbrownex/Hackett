''' Prepare the data for modeling:
    - Split X and Y
    - Format X
    '''
__author__ = "Tom Browne"

maxRGB = 255

def reshape(X):
    ''' Keras requires a 4-dimensional array as input '''
    return X.reshape(X.shape[0], 28, 28, 1)

def retype(X):
    ''' Make sure we get a decimal result from division '''
    return X.astype('float32')

def normalize(X):
    ''' Divide by max RGB value '''
    X /= maxRGB
    return X

def formatX(X):
    X = reshape(X)
    X = retype(X)
    X = normalize(X)
    return X

def preProcess(train, test, config, args):
    ''' "train" and "test" are tuples of X and Y '''
    trainX, trainY = train
    testX, testY   = test
    trainX         = formatX(trainX)
    testX          = formatX(testX)
    
    dataDict    = {}
    dataDict["trainX"] = trainX
    dataDict["trainY"] = trainY
    dataDict["testX"]  = testX
    dataDict["testY"]  = testY
    
    return dataDict