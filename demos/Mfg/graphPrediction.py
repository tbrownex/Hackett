import pandas as pd
import numpy  as np
import sys
import random
import matplotlib.pyplot as plt

from getConfig import getConfig

def plot(df, unit):
    actuals = df["actual"]
    actuals = actuals.values.reshape([-1,1])
    predictions = df["ensemble"]
    predictions = predictions.values.reshape([-1,1])
    plt.plot(actuals, label="actual")
    plt.plot(predictions, label="prediction")
    plt.legend(loc='upper right')
    unit = "Unit # " + str(unit)
    plt.annotate(unit, xy=(0.05, 0.05), xycoords='axes fraction')
    plt.savefig(config["JSONloc"]+'image.png', bbox_inches='tight')

if __name__ == "__main__":
    config = getConfig()
    df = pd.read_csv(config["JSONloc"]+"predictions.csv")
    df.set_index("unit", inplace=True)
    units = df.index.values
    
    if len(sys.argv) > 1:
        unit = int(sys.argv[1])
    else:
        unit = np.random.choice(units, 1)

    df = df.loc[unit]
    plot(df, unit)
    