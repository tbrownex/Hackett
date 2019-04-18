''' Prepare the data for modeling:
    - Split X and Y
    - Format X
    '''
__author__ = "Tom Browne"

import numpy as np
from scipy import ndimage

maxRGB = 255

def adjustImage(X):
    X.setflags(write=1)
    ''' all these character images are mirror-imaged and rotated so to get them right-side up 
    need to flip along vertical axis then rotate clockwise '''
    angle = -90
    bg_value = -0.5       # this is regarded as background's value black
    for n in range(X.shape[0]):
        adjusted = np.flip(X[n], axis=0)
        X[n] = ndimage.rotate(adjusted, angle, reshape=False, cval=bg_value)
    return X

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
    X = adjustImage(X)
    X = reshape(X)
    X = retype(X)
    X = normalize(X)
    return X

def preProcess(train, test, config):
    ''' "train" and "test" are tuples of X and Y '''
    trainX, trainY = train
    testX, testY   = test
    trainX         = formatX(trainX)
    testX          = formatX(testX)
    
    # The labels run from 1-26 and we need them 0-25 (index of the output layer) so subtract 1
    trainY = np.subtract(trainY, 1)
    testY = np.subtract(testY, 1)
    
    dataDict    = {}
    dataDict["trainX"] = trainX
    dataDict["trainY"] = trainY
    dataDict["testX"]  = testX
    dataDict["testY"]  = testY
    
    return dataDict