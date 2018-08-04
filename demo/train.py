import pandas as pd
import numpy  as np
import os
import sys
import time

from scale_split_data import partition
from getDataDirectory import getDir
from nn               import run
import jobNumber      as job
from split_label      import splitY

TEST_PCT = 0
VAL_PCT  = 0.20
FILENM   = 'HH+neighborhood_rank.csv'

# NN hyper-parameters
l1_size       = [32]          # Count of nodes in layer 1
learning_rate = [.00001]
Lambda        = [0.02]          # Regularization parameter
weight        = [1]              # Degree to which Positives are weighted in the loss function
batch_size    = [512]
epochs        = [5]
activation    = ['ReLU']           # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'

def getData():
    data_dir = "/home/tbrownex/data/Hackett/"
    print("reading data")
    cols = ["Label","age_alt","income_alt","marital_status","owner_type",\
            "home_value","num_hh","num_adults","num_kids","net_worth_alt","density",\
            "age_bins","income_bins","hv_bins","density_bins","white","black",\
            "hispanic","asian","jewish","indian","other","neighborhood_bin"]
    df = pd.read_csv(data_dir+FILENM, sep="|", usecols=cols,nrows=4000000)
    print("done")
    df       = df.sample(frac = 1.0)
    
    # Drop any columns with NAN values
    before = df.shape[0]
    df     = df.dropna(axis=0, how='any')
    after  = df.shape[0]
    print("{}Deleted {:,.0f} rows out of {:,.0f} due to NAN".format(
            "\n",(before-after), before))
    return(df)

def prepareData():
    '''Read the training data file and create a dictionary with keys of "train_x", "train_labels",
    "val_x" and "val_labels"'''
    df = getData()
    # Move the label to the last column
    df["label"] = df["Label"]
    del df["Label"]
    train, val, _ = partition(df, VAL_PCT, TEST_PCT)
    data_dict     = splitY(train, val, None)
    return data_dict

def print_sales_ratio(data_dict):
    pos = np.sum(data_dict['train_labels'][:,1])
    tot = data_dict['train_labels'].shape[0]
    print("{}{:>18} {:>12,.0f} rows with {:,.0f} Positives  {}:1".format(
            "\n", "Training file:",tot, pos, int(tot/pos)))
    pos = np.sum(data_dict['val_labels'][:,1])
    tot = data_dict['val_labels'].shape[0]
    print("{:>18} {:>12,.0f} rows with {:,.0f} Positives  {}:1{}".format("Validation file:",
            tot, pos, int(tot/pos), "\n"))
    input()
# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results, job_id):
    with open("/home/tom/summary_"+str(job_id)+".txt", 'w') as summary:
        keys = results[0][1]
        hdr = "Run" +"|" + "|".join(keys)
        hdr += "|"+"Lift" + "\n"
        summary.write(hdr)        
        
        for x in results:
            rec = str(x[0]) +"|"
            rec += "|".join([str(t) for t in x[1].values()])
            rec += "|"+ str(x[2]) +"\n"         # lift
            summary.write(rec)
            
if __name__ == "__main__":
    data_dict = prepareData()    
    print_sales_ratio(data_dict)
    
    job_id = job.getJob()
    
    parm_dict = {}                  # holds the hyperparameter combination for one run
    count = 1
    parm_dict['l1_size']       = l1_size[0]
    parm_dict['lambda']        = Lambda[0]
    parm_dict['weight']        = weight[0]
    parm_dict['batch_size']    = batch_size[0]
    parm_dict['epochs']        = epochs[0]
    parm_dict['activation']    = activation[0]
    parm_dict['learning_rate'] = .00003
    parm_dict['std']           = 1.0
    job_name = "job_" + job_id +"/"+ "run_" + str(count)
    
    run(data_dict, parm_dict, job_name)
    
    count +=1
    parm_dict['learning_rate'] = .00003
    parm_dict['std']           = 0.5
    job_name = "job_" + job_id +"/"+ "run_" + str(count)
    
    run(data_dict, parm_dict, job_name)
    
    count +=1
    parm_dict['learning_rate'] = .00001
    parm_dict['std']           = 0.25
    job_name = "job_" + job_id +"/"+ "run_" + str(count)
    
    run(data_dict, parm_dict, job_name)
    
    job_id = int(job_id)
    job.setJob(job_id+1)