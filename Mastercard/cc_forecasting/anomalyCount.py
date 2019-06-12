# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:58:39 2018

@author: William Cairns
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from metricsCollector import listener

def anomalyModel(df,config):
    
    listDriver = df.Driver.unique()
    U = df.UUID.unique()
    
    data = []
    for Driver in listDriver:
        driverDf = df.loc[df['Driver']==Driver]
        
        driverDf = driverDf.drop('Driver',axis=1)
        driverDf = driverDf.drop('Date',axis=1)

        df_num = StandardScaler().fit_transform(driverDf)

        A = []
        B = []
        C = []

        for i in np.linspace(0.1,5,50):
            db = DBSCAN(eps=i, min_samples=10).fit(df_num)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
            sum = 0
            for t in labels:
                if t == -1: 
                    sum = sum + 1
            C.append(sum)
            
            
            A.append(i)
            B.append(int(n_clusters_))

        results = pd.DataFrame([A,B,C]).T
        results.columns = ['distance','Number of clusters','Number of outliers']

        driverDf = driverDf.join(pd.DataFrame(labels))
        driverDf = driverDf.rename(columns={0:'Cluster'})
        
        z= driverDf[driverDf.Cluster == -1].count()
        data.append(z.Cluster)    
    

    metric = ['anomaly'] * len(data)
    dfOutput = pd.DataFrame([metric, data]).T
    dfOutput.insert(loc=0,column='UUID',value=U)
    dfOutput.insert(loc=1,column='Driver',value=listDriver)
    dfOutput.columns = ['UUID','Driver', 'Metric', 'Value']

    listener(dfOutput)
    
    return(df)
    