'''
    A dictionary object holds key parameters such as:
    baseDir - Where the input data is located 
    fileName - filename of the input file
    outputDir - location to write the model prediction file
    logLoc - Log file directory
    logFile - Log file name
    logDefault - Logging level (info, warn, error)
    lossMonths - Number of flat or silent periods to qualify as a Loss (right censored)
    winMonths - Number of flat or silent periods to qualify as a Win (left censored)
    censorFile - Output filename to catalog wins and losses
    Test - For development purposes, allows certain features to be turned off 
    filterDriver - For development purposes, allows the input data to be filtered 
    inSamplePeriods - Number of Periods that would be used for 'test' vs. 'train'
    outSamplePeriods - Number of future periods that would be used to predict
    paramDir - ARIMA Parameters directory
    paramFile - ARIMA Parameters Filename
'''

__author__ = 'The Hackett Group'

def getConfig():

    d = {}
    d['baseDir']    = 'C:/Users/e084332/Documents/RevenueForecasting/'
    d['fileName']  = 'sample_5Nov2018.csv'
    d['outputDir']    = 'C:/Users/e084332/Documents/RevenueForecasting/output/'
    d['logLoc']     = 'H:/sandbox'
    d['logFile']    = 'logstash.log'
    d['logDefault'] = 'info'
    d['lossMonths'] = 7
    d['winMonths']  = 12
    d['censorFile'] = 'WinsLosses.csv'
    d['Test']       = False
    d['filterDriver'] = 'finance'
    d['inSamplePeriods'] = '13'
    d['outSamplePeriods'] = '13'
    d['paramDir'] = 'C:/Users/e084332/Documents/RevenueForecasting/ArimaParams/'
    d['paramFile']  = 'ProcessedVolume25_30Oct2018.csv'
    return d
