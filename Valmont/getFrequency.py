def getFrequency(col):
    '''
    Determine whether the MEI column is quarterly or monthly based on how many unique values:
    Quarterly has its values repeated 3 times for the months in a quarter so will have 1/3 the number
    of unique values
    '''
    s = col.unique()
    if len(s) > 25:
        return "Monthly"
    else:
        return "Quarterly"