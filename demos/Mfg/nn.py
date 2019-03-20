import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import sys
import time

EPOCHS = 40
    
def runNN(dataDict, parmDict, svUnits, config):
    '''data: dictionary holding Train and Validation sets'''
    feature_count = dataDict['trainX'].shape[1]
    # Load hyper-parameters
    L1         = parmDict['l1Size']
    LR         = parmDict['lr']
    #LAMBDA     = parmDict['lambda']
    BATCH      = parmDict['batchSize']
    ACTIVATION = parmDict['activation']
    KEEP       = 1 - parmDict['dropout']
    STD        = parmDict["std"]
    
    # Set up the network
    tf.reset_default_graph()
    x  = tf.placeholder("float", shape=[None, feature_count], name="input")
    y_ = tf.placeholder("float", shape=[None,])
    
    # Layer 1
    l1_w     = tf.Variable(tf.truncated_normal([feature_count, L1], stddev=STD, dtype=tf.float32))
    l1_b     = tf.Variable(tf.zeros([1,L1], dtype=tf.float32))
    if parmDict['activation'] == "tanh":
        l1_out = tf.nn.tanh(tf.matmul(x,l1_w) + l1_b)
    elif parmDict['activation'] == "relu":
        l1_out = tf.nn.relu(tf.matmul(x,l1_w) + l1_b)
    else:
        print("Invalid activation...shutting down: ", parmDict["activation"])
        sys.exit()
    l1_drop = tf.nn.dropout(l1_out,\
                            keep_prob=KEEP)
    
    initializer = tf.contrib.layers.xavier_initializer()
    l2_w   = tf.Variable(initializer([L1,1]))
    l2_b   = tf.Variable(tf.zeros([1,1]))
    l2_out = tf.add(tf.matmul(l1_drop, l2_w), l2_b, name="L2")
    
    # Cost function
    cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(l2_out, y_)))
    '''LAMBDA = 0.01
    A = LAMBDA**tf.reduce_sum(tf.abs(l1_w))
    B = LAMBDA*tf.nn.l2_loss(l1_w)
    C = LAMBDA*tf.nn.l2_loss(l2_w)
    cost = tf.reduce_mean(mse + A + B + C)'''
    
    # Optimizer
    optimize = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)
        
    valCost = tf.summary.scalar('Validation cost', cost)
    merged = tf.summary.merge_all()

    # Run
    TB_counter = 0                    # For TensorBoard
    num_training_batches = int(len(dataDict['trainX']) / BATCH)
    print('{} epochs of {} iterations with batch size {}'.format(EPOCHS,num_training_batches,BATCH))
    
    saver = tf.train.Saver()
    
    #CP = tf.ConfigProto( device_count = {'GPU': 1} )
    #sess = tf.Session(config=CP)
    
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(config["TBdir"] + '/tom', sess.graph)
    sess.run(tf.global_variables_initializer())
    
    for i in range(EPOCHS):
        a,b = shuffle(dataDict['trainX'],dataDict['trainY'])
        for j in range(num_training_batches):
            x_mini = a[j*BATCH:j*BATCH+BATCH]
            y_mini = b[j*BATCH:j*BATCH+BATCH]
            _ = sess.run(optimize, feed_dict = {x: x_mini, y_: y_mini})
        if i %5==0:
            epochMSE = sess.run(cost, feed_dict = {x: dataDict['valX'], y_: dataDict['valY']})
            print(i, ": ", epochMSE)
            #train_writer.add_summary(vc, TB_counter)
            #TB_counter += 1
            
    preds = sess.run(l2_out, feed_dict = {x: dataDict['testX']})
    preds = np.reshape(preds, newshape=[-1,])
    modelName = config["modelDir"] + "NNmodel_1"
    saver.save(sess, modelName)
    train_writer.close()
    return preds