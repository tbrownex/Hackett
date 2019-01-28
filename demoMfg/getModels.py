import glob

def getModels(typ, job, config):
    '''
    Models have been trained and saved. Get all the models, depending on the typ
    '''
    assert (typ in ["RF", "XGB", "NN"]), "invalid Model Type"
    
    if typ == "RF":
        m = config["modelDir"]+"RFmodel*"
    elif typ == "NN":
        m = config["modelDir"]+"NNmodel_" + job + "*.meta"
    
    return glob.glob(m)