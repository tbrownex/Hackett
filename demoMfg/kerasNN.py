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

# TODO:
#    add Cross Validation
#    add Early Stopping
#    view Feature Importance

def buildLayers(parmDict):
    Dense   = partial(keras.layers.Dense)
    Dropout = partial(keras.layers.Dropout)
    
    nn = tf.keras.Sequential()
    nn.add(Dense(parmDict["l1Size"],\
                 kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=parmDict["std"]),\
                                                                       activation=parmDict["activation"]))
    nn.add(Dropout(parmDict["dropout"]))
    nn.add(Dense(1, activation="linear"))
    return nn

def buildNetwork(parmDict):
    nn = buildLayers(parmDict)
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
    TB        = keras.callbacks.TensorBoard(log_dir=config["TBdir"])
    '''modelSave = keras.callbacks.ModelCheckpoint(config["modelDir"],
                                                save_weights_only=True,
                                                verbose=1)'''
    
    X = np.array(dataDict["trainX"])
    Y = np.array(dataDict["trainY"])
    nn.fit(X, Y,\
           batch_size=parmDict["batchSize"],\
           epochs=40,\
           validation_split=0.15,\
           verbose=0,\
           shuffle=False,
           callbacks=[TB])

def formatPreds(dataDict, svUnits, preds):
    ''' Prepare the data to be evaluated '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["pred"]   = np.reshape(preds, [-1,])
    d["unit"]   = svUnits
    df = pd.DataFrame(d)
    df.set_index("unit", inplace=True)
    return df

def runNN(dataDict, parmDict, svUnits, config):    
    '''data: dictionary holding Train, Validation and Test sets'''
    nn = buildNetwork(parmDict)
    fitNetwork(dataDict, parmDict, nn, config)
    tf.keras.models.save_model(model=nn, filepath=config["modelDir"]+"NNmodel.h5")
    preds = nn.predict(dataDict["testX"])
    df    = formatPreds(dataDict, svUnits, preds)
    mape, rmse = evaluate(df, config["evaluationMethod"])
    return mape, rmse