import tensorflow as tf

__author__ = "Tom Browne"

def getData():
    ''' Keras has a function to return the downloaded data '''
    train, test = tf.keras.datasets.mnist.load_data()
    return train, test