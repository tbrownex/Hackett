''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - an indicator allowing execution in Test mode'''

__author__ = "The Hackett Group"

def getConfig():
    d = {}
    d["bucketName"] = "ml-datasets1"
    d["fileNames"] = ["EMNIST/emnist-letters-test-images-idx3-ubyte",
                      "EMNIST/emnist-letters-test-labels-idx1-ubyte",
                      "EMNIST/emnist-letters-train-images-idx3-ubyte",
                      "EMNIST/emnist-letters-train-labels-idx1-ubyte"]
    d["inputShape"] = [28,28,1]
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "EMNIST.log"
    d["logDefault"] = "info"
    d["TBdir"] = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/TF/models/"  # where to save models
    return d