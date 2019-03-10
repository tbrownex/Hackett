import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from getArgs    import getArgs
from getConfig  import getConfig
from getData    import getData
from preProcess import preProcess

__author__ = "Tom Browne"

def createNetwork(inputShape):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    return model

def fitNetwork(dataDict, model, config):
    TB = keras.callbacks.TensorBoard(log_dir=config["TBdir"])
    filepath = config["modelDir"]+"NNmodel.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath,\
                                                 save_weights_only=False,\
                                                 monitor='val_acc',\
                                                 verbose=1,\
                                                 save_best_only=True,\
                                                 mode='max')

    model.fit(dataDict["trainX"], dataDict["trainY"],\
              #batch_size=parmDict["batchSize"],\
              epochs=3,\
              validation_split=0.15,\
              verbose=0,\
              shuffle=False,
              #callbacks=[TB],
              callbacks=[checkpoint])

def printResults(results, model):
    print("\n{:<10}{}".format("Metric", "Value"))
    for idx, metric in enumerate(model.metrics_names):
        print("{:<10}{:.2f}".format(metric, results[idx]))

def process(dataDict, config):
    model = createNetwork(config["inputShape"])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    fitNetwork(dataDict, model, config)
    print("\nRunning Test data...")
    results = model.evaluate(dataDict["testX"], dataDict["testY"])
    printResults(results, model)

if __name__ == "__main__":
    args   = getArgs()
    config = getConfig()
    train, test = getData()
    dataDict = preProcess(train, test, config, args)
    
    print("\nTraining with {:,} images of shape {}".format(dataDict["trainX"].shape[0],\
                                                         dataDict["trainX"][0].shape))
    print('Testing with {:,} images'.format(dataDict["testX"].shape[0]))
    process(dataDict, config)