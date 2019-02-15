import numpy as np

# Apply the weights and calculate a score


def scoreRow(row, weights):
    score = (row*weights).sum()
    return score

def createLabel(df):
    colCount = df.shape[1]
    # Assign weights to each attribute
    weights = np.array([np.random.uniform(low=-2.5, high=4.0) for x in range(colCount)])

    df["Score"] = df.apply(scoreRow, args=(weights,), axis=1)
    return df