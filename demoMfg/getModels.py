import glob

def getModels(typ, config):
    '''
    Models have been trained and saved. Get all the models for the "typ" passed
    '''
    assert (typ in ["RF", "XGB", "NN"]), "invalid Model Type"
    
    if typ == "RF":
        m = config["modelDir"]+"RFmodel*"
    elif typ == "NN":
        m = config["modelDir"]+"NN*"
    
    return glob.glob(m)