import idx2numpy

from copyFiles import copyFiles

PATH = "/tmp/"

def loadFile(fileName):
    ''' This takes a 'bytes' object and converts it to np array '''
    return idx2numpy.convert_from_file(PATH+fileName)

def getData(config):
    ''' Copy the files from Cloud to local, then load them into arrays '''
    copyFiles(config)
    
    fileName = "emnist-letters-train-images-idx3-ubyte"
    trainImages = loadFile(fileName)
    fileName = "emnist-letters-train-labels-idx1-ubyte"
    trainLabels = loadFile(fileName)
    
    fileName = "emnist-letters-test-images-idx3-ubyte"
    testImages = loadFile(fileName)
    fileName = "emnist-letters-test-labels-idx1-ubyte"
    testLabels = loadFile(fileName)
    
    
    train = (trainImages, trainLabels)
    test = (testImages, testLabels)    
    return train, test