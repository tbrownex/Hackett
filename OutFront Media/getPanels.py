from os import listdir
from os.path import isfile, join


def getPanels(dataloc):    
    panels = [f for f in listdir(dataloc+"panels/") if isfile(join(dataloc+"panels/", f))]
    print("{:,.0f} panels to process".format(len(panels)))
    return panels