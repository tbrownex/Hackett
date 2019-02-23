import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

EPOCHS = 100

def run(dataDict, parms, config, jobName):
    '''data: dictionary holding Train and Validation sets'''
    featureCount = dataDict['trainX'].shape[1]
    # Load hyper-parameters
    L1         = parms['l1Size']
    LAMBDA     = parms['lambda']
    ACTIVATION = parms['activation']
    BATCH      = parms['batchSize']
    LR         = parms['lr']
    STD        = parms['std']
    
    # Set up the network
    tf.reset_default_graph()
    x  = tf.placeholder("float", shape=[None, featureCount], name="input")
    y_ = tf.placeholder("float", shape=[None,], name="label")

    l1_w     = tf.Variable(tf.truncated_normal([featureCount, L1], stddev=STD, dtype=tf.float32, seed=1814))
    l1_b     = tf.Variable(tf.truncated_normal([1,L1], dtype=tf.float32))
    
    if   ACTIVATION == 'tanh':
        l1_act = tf.nn.tanh(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'leakyReLU':
        l1_act   = leakyReLU(x, l1_w, l1_b)
    elif ACTIVATION == 'ReLU':
        l1_act   = tf.nn.relu(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'ReLU6':
        l1_act   = tf.nn.relu6(tf.matmul(x,l1_w) + l1_b)
        
    l2_w   = tf.Variable(tf.truncated_normal([L1,1], stddev=STD, dtype=tf.float32, seed=1814))
    l2_b   = tf.Variable(tf.truncated_normal([1,1]))

    l2_out = tf.add(tf.matmul(l1_act, l2_w), l2_b, name="L2")
    
    # Cost function
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(l2_out, y_)))
    L1_layer1 = LAMBDA*tf.reduce_sum(tf.abs(l1_w))
    L2_layer1 = LAMBDA*tf.nn.l2_loss(l1_w)
    L2_layer2 = LAMBDA*tf.nn.l2_loss(l2_w)
    
    cost = tf.reduce_mean(rmse + L1_layer1 + L2_layer1 + L2_layer2)
    
    # Optimizer
    optimize = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)
        
    valCost = tf.summary.scalar('Validation cost', rmse)
    merged = tf.summary.merge_all()

    # Run
    TBcounter = 1                    # For TensorBoard
    num_training_batches = int(len(dataDict['trainX']) / BATCH)
    
    saver = tf.train.Saver()
    
    #CP = tf.ConfigProto( device_count = {'GPU': 1} )
    #sess = tf.Session(config=CP)
    
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(config["TBdir"] + "/" + jobName, sess.graph)
    sess.run(tf.global_variables_initializer())
    
    val_score_best = early_stop_counter = 0
    
    for i in range(EPOCHS):
        if early_stop_counter > 5000:
            print("early stop")
            break
        a,b = shuffle(dataDict['trainX'],dataDict['trainY'])
        costList = []
        for j in range(num_training_batches):
            x_mini = a[j*BATCH:j*BATCH+BATCH]
            y_mini = b[j*BATCH:j*BATCH+BATCH]
            _, tc = sess.run([optimize, cost], feed_dict = {x: x_mini, y_: y_mini})                
        vc = sess.run(merged,
                      feed_dict = {x: dataDict['valX'],
                                   y_: dataDict['valY']})
        #train_writer.add_summary(vc, TBcounter)
        TBcounter += 1
        '''if val_score > val_score_best * 1.1:
                    val_score_best = val_score
                    early_stop_counter = 0
                else:
                    if val_score_best > 2.0:
                        early_stop_counter += 1'''
    preds = sess.run(l2_out,
                     feed_dict = {x: dataDict['testX'],
                                  y_: dataDict['testY']})
    #saver.save(sess, SAVE_DIR+'SS_'+job_name )
    train_writer.close()
    return preds