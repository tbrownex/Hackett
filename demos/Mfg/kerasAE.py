from functools import partial
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
import itertools
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from evaluate import evaluate
from calcRMSE import calcRMSE
from calcMAPE import calcMAPE

def buildLayers(parmDict, featureCount):
    Dense   = partial(keras.layers.Dense)
    Dropout = partial(keras.layers.Dropout)
    
    nn = tf.keras.Sequential()
    nn.add(Dense(parmDict["l1Size"],\
                 kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=parmDict["std"]),\
                                                                       activation=parmDict["activation"]))
    nn.add(Dropout(parmDict["dropout"]))
    nn.add(Dense(parmDict["l2Size"],\
                 kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=parmDict["std"]),\
                                                                       activation=parmDict["activation"]))
    nn.add(Dropout(parmDict["dropout"]))
    nn.add(Dense(featureCount, activation="linear"))
    return nn

def buildNetwork(parmDict, featureCount):
    nn = buildLayers(parmDict, featureCount)
    if parmDict["optimizer"] == "SGD":
        opt  = tf.keras.optimizers.SGD(lr=parmDict["lr"],\
                                       decay=1e-6,\
                                       momentum=0.0,
                                       nesterov=True)
    elif parmDict["optimizer"] == "Adam":
        opt = tf.keras.optimizers.Adam(lr=parmDict["lr"],\
                                       beta_1=0.9,\
                                       beta_2=0.999,\
                                       epsilon=None,\
                                       decay=1e-6,\
                                       amsgrad=False)
    nn.compile(optimizer=opt, loss="mse")
    return nn

def fitNetwork(dataDict, parmDict, nn, config):
    TB = keras.callbacks.TensorBoard(log_dir=config["TBdir"])
    
    X = np.array(dataDict["trainX"])
    nn.fit(X, X,\
           batch_size=parmDict["batchSize"],\
           epochs=40,\
           validation_split=0.15,\
           verbose=0,\
           shuffle=False,
           callbacks=[TB])

def scoreNetwork(df, nn):
    data  = np.array(df)
    preds = nn.predict(x=data)
    mse  = (np.square(data - preds)).mean(axis=None)
    mape = np.mean(np.abs((data - preds) / data))
    return mape, mse

def runNN(dataDict, parmDict, config):    
    '''data: dictionary holding Train, Validation and Test sets'''
    featureCount = dataDict["trainX"].shape[1]
    
    nn = buildNetwork(parmDict, featureCount)
    fitNetwork(dataDict, parmDict, nn, config)
    tf.keras.models.save_model(model=nn, filepath=config["modelDir"]+"AEmodel.h5")
    scores = scoreNetwork(dataDict["testX"], nn)
    return scores