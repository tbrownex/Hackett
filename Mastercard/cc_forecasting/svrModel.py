
# coding: utf-8

"""
# # Detrended Z-Score Hybrid SVR Model
# Depreciated this module 10/27/2018
"""
# Imports
import numpy as np
import pandas as pd
import dateutil
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from datetime import datetime,date
from abstractWriter import csvWriter
from getConfig import getConfig


def trainSVR(df,config):

    
    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,decimals = 0)

    #df['Date'] = df['Date'].apply(dateutil.parser.parse, dayfirst=False)
    dfOutput = pd.DataFrame()
    uuid = []
    driver = []
    mape = []

    listDriver = df.Driver.unique()
    
    for driverTemp in listDriver:
        driverDf = df[df['Driver'] == driverTemp].copy()
        
        # First two columns for the output file
        uuid.append(str(driverDf.iloc[0,2]))
        driver.append(driverTemp)
        

        # Find the earlest date of the data which will be used to create
        # the new column 'elapased_months'.
        # This will be modified to take from config in the future
        minDate = df['Date'].min()

        # Here is the elapased months logic. Subtract the minimum then add 1
        driverDf['elapased_months'] =  (driverDf.Date.dt.to_period('M') - minDate.to_period('M'))+1

        # Python issue: elapased months is type 'object' what I need is integer
        driverDf['elapased_months'] = driverDf['elapased_months'].astype(str).astype(int)

        # Set the index which allows easy subsetting into train and test sets
        driverDf = driverDf.set_index('Date')

        # Holdout the last year for testing
        train = driverDf.loc['2004-01':'2017-07'].copy()
        test = driverDf.loc['2017-08':].copy()

        # Detrend the training data
        # returns a new column named 'detrended'
        # Elapased months converted to an array
        X_train = train['elapased_months'].tolist()
        X_train = np.reshape(X_train,(len(X_train), 1))
    
        # Run the Model
        lrmodel = LinearRegression()
        lrmodel.fit(X_train, train.y)

        # calculate trend
        trend = lrmodel.predict(X_train)

        # detrend the training data
        train['detrended'] = [train.y[i]-trend[i] \
            for i in range(0, len(train.y))]

        slope = lrmodel.coef_[0]
        intercept = lrmodel.intercept_
        #print('Slope for the trained model is: {}'.format(slope))
        #print('Intecept for the trained model is: {}'.format(intercept))
        
        # Z-Score Adjust the training data
        # returns a new column named 'z_train'
        z_sigma = train['detrended'].std(ddof=0)
        z_hat = train['detrended'].mean()
        train['z_train'] = (train['detrended'] - z_hat)/z_sigma

        #train.head()

        # ## ------------- SVR Model Training --------------
        # The model will take lists so a simple conversion
        X_train = train['elapased_months'].tolist()
        Z_train = train['z_train'].tolist()
        X_test = test['elapased_months'].tolist()

        # Elapased months converted to an array
        X_train = np.reshape(X_train,(len(X_train), 1))

        # Model object with parameters for tuning
        svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)

        # Train the model
        svr_rbf.fit(X_train, Z_train)

        # Predict the last 14 months by giving a vector of months
        z_pred = svr_rbf.predict(np.reshape(X_test,(len(X_test), 1))).tolist()

        # Rescale the affect removed by the 
        # z-score adjustment to the training data
        pred = [(z_pred[i]*float(z_sigma)) + z_hat \
            for i in range(0, len(z_pred))]

        # Get the trend from the linear model
        z_trend = lrmodel.predict(np.reshape(X_test,(len(X_test), 1))).tolist()

        test = test.reset_index()

        testOutput = pd.DataFrame([pred,z_trend,test.y.tolist()]).T
        testOutput.columns = ['Prediction', 'linPrediction', 'obsValue']

        testOutput['adjPrediction'] = testOutput['Prediction'] + testOutput['linPrediction']

        #print(str(driverDf.iloc[0,1]))
        #print(str(driverTemp))
        
        #print(mean_absolute_percentage_error(testOutput['obsValue'],testOutput['adjPrediction']))

        mape.append(mean_absolute_percentage_error(testOutput['obsValue'],testOutput['adjPrediction']))


    dfOutput = pd.DataFrame([uuid,driver,mape]).T
    dfOutput.columns = ['UUID', 'Driver','MAPE']
    
    csvWriter(dfOutput,'SVRModel',config)


    return df

