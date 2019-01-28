import h2o

def getData(config):
    typ = {"Pool": "string"}
    return h2o.import_file(path = config["dataLoc"] + config["fileName"], col_types=typ)