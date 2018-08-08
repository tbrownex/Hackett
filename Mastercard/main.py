import sys
import pandas as pd
from config import getClient
from checkMissing import checkMissing
from imputation import impute

TEST_WEEKS = 6
stats = {}

def getData(country):
    dataloc = getClient("MC")
    df      = pd.read_csv(dataloc+"Processed.csv")
    del df["index"]
    df["Month"] = pd.to_datetime(df['Month'])
    df      = df.set_index("Country")
    df      = df.loc[country]
    df      = df.reset_index(drop=True)
    df      = df.set_index(["Program", "Customer", "Driver"])
    df      = df.sort_values(["Program", "Customer", "Driver", "Month"])
    return df

def imputation(df):
    from random import randint
    dataloc = getClient("MC")       # get the location to write the updated file
    svDF = df.copy()
    missing = checkMissing(df)
    if missing:
        stats["imputed"] += 1       # "Needs imputation" counter
        complete, df = impute(df)
        if not complete:
            stats["incomplete"] += 1  # "Not enough data to bother imputing" counter
        else:
            test = randint(1, 11)
            if test == 7:
                svDF.to_csv(dataloc+"test_before.csv", sep="|")
                df.to_csv(dataloc+"test_after.csv", sep="|")
    return df
            
# Create Train and Test sets
def split(val):
    amounts = val["Amount"]
    amounts = amounts.reset_index(drop=True)
    dates   = val["Month"]
    dates   = dates.reset_index(drop=True)
    
    # Create Train and Test sets
    train = {}
    test = {}
    train["Amount"] = amounts[:-TEST_WEEKS]
    test["Amount"]  = amounts[-TEST_WEEKS:]
    train["Dates"]  = dates[:-TEST_WEEKS]
    test["Dates"]   = dates[-TEST_WEEKS:]
    
    # Make sure splitting train & test didn't lose any data
    assert (amounts.sum() - train["Amount"].sum() - test["Amount"].sum() < 0.1 )
    return train, test

# Input is a dataframe for a single country, sorted by PKID and month
# Loop over each PKID, which is the index to the df
def process(df):
    stats["count"] = 0
    stats["incomplete"] = 0
    stats["imputed"] = 0
    for idx, val in df.groupby(level=[0, 1, 2]):
        stats["count"] += 1
        pkid = "|".join(idx)
        val = imputation(val)
      
    print("Total PKIDs processed: {:,.0f}".format(stats["count"]))
    print("Number needing imputation: {:,.0f}, {:.1%}".\
          format(stats["imputed"],stats["imputed"]/stats["count"]))
    print("Number incomplete: {:,.0f}".format(stats["incomplete"]))
        
        #train, test = split(val)     # these are now single streams to forecast
        #return train, test
        #print(train)
        #print(test)
        #input()

if __name__ == "__main__":
    country = sys.argv[1]
    df = getData(country)
    process(df)