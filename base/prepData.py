''' Prepare the data for modeling:
    
    - Create the Label
    - Create Train and Test sets
    - Split the features from the labels
    
    - "df" has the incoming data
    - "config" is a data dictionary with configuration parameters'''

from sklearn.model_selection import train_test_split
import createLabel

__author__ = "The Hackett Group"

def splitLabels(train, test, config):
    # Separate the features and labels
    d = {}
    d["trainY"] = train[config["labelColumn"]]
    del train[config["labelColumn"]]
    d["trainX"] = train
    
    d["testY"] = test[config["labelColumn"]]
    del test[config["labelColumn"]]
    d["testX"] = test
    return d

def process(df, config):
    df = createLabel.calcMean(df, config)
    train, test = train_test_split(df, test_size=config["testPct"])
    d = splitLabels(train, test, config)
    return d