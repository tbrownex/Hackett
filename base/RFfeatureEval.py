''' For a Random Forest, show the most important features

"cols" is a list of all the df columns
"model" is the RandomForestRegressor model'''

import numpy as np

def process(cols, model):
    tmp = zip(cols, model.feature_importances_)
    features = sorted(tmp, key=lambda tup: tup[1], reverse=True)

    print("{:<15}{}".format("Feature", "importance"))
    for x in features:
        print("{:<15}{:.2f}".format(x[0], x[1]))
    return features