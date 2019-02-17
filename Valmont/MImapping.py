def MImapping():
    d = {}

    # GDP
    tmp = {}
    tmp["freq"] = "Q"
    tmp["col"]  = "GDP"
    d["GDP.csv"] = tmp
    # Unemployment Rate
    tmp = {}
    tmp["freq"] = "M"
    tmp["col"]  = "unempRate"
    d["unempRate.csv"] = tmp
    return d