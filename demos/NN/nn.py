import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import sys
import time

EPOCHS = 10

def run(dataDict, parmDict, config):    
    featureCount = dataDict['trainX'].shape[1]
    
    # Load hyper-parameters
    L1         = parmDict['l1Size']
    ACTIVATION = parmDict['activation']
    LAMBDA     = parmDict['Lambda']
    BATCH      = parmDict['batchSize']
    LR         = parmDict['lr']
    STD        = parmDict["std"]
    DROP       = parmDict["dropout"]
    OPT        = parmDict["optimizer"]
    WEIGHT     = parmDict['weight']
    
    # Set up the network
    tf.reset_default_graph()
    x  = tf.placeholder("float", shape=[None, featureCount], name="input")
    y_ = tf.placeholder("float", shape=[None, config["numClasses"]])

    l1_w = tf.Variable(tf.truncated_normal([featureCount, L1], stddev=STD, dtype=tf.float32, seed=1814))
    l1_b = tf.Variable(tf.truncated_normal([1,L1], dtype=tf.float32))
    
    if   ACTIVATION == 'tanh':
        l1_act = tf.nn.tanh(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'leakyReLU':
        l1_act   = leakyReLU(x, l1_w, l1_b)
    elif ACTIVATION == 'ReLU':
        l1_act   = tf.nn.relu(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'ReLU6':
        l1_act   = tf.nn.relu6(tf.matmul(x,l1_w) + l1_b)
        
    l2_w   = tf.Variable(tf.truncated_normal([L1,config["numClasses"]], stddev=STD, dtype=tf.float32, seed=1814))
    l2_b   = tf.Variable(tf.truncated_normal([1,config["numClasses"]]))

    l2_out = tf.add(tf.matmul(l1_act, l2_w), l2_b, name="L2")
    
    # Cost function
    mse = tf.losses.mean_squared_error(y_, l2_out)
    
    # Optimizer
    optimize = tf.train.AdamOptimizer(learning_rate=LR).minimize(mse)

    num_training_batches = int(len(dataDict['trainX']) / BATCH)
    #print('{} epochs of {} iterations with batch size {}'.format(EPOCHS,num_training_batches,BATCH))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print("{:<8}{}".format("Epoch", "ValCost"))
    for i in range(EPOCHS):
        a,b = shuffle(dataDict['trainX'],dataDict['trainY'])
        for j in range(num_training_batches):
            x_mini = a[j*BATCH:j*BATCH+BATCH]
            y_mini = b[j*BATCH:j*BATCH+BATCH]
            _, error = sess.run([optimize, mse], feed_dict = {x: x_mini, y_: y_mini})
        valCost = sess.run(mse, feed_dict = {x: dataDict["valX"],
                                             y_:dataDict["valY"]})
        print("{:<8}{:<.4f}".format(i, valCost))
    final, preds = sess.run([mse, l2_out], feed_dict = {x: dataDict["testX"],
                                       y_:dataDict["testY"]})
    np.savetxt("/home/tbrownex/preds.csv", preds)
    np.savetxt("/home/tbrownex/testY.csv", dataDict["testY"])
    
    
    return final