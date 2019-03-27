import csv

def getFrequency():
    '''
    colFrequency has been loaded with a mapping of Column:Frequency (Weekly, monthly...)
    '''
    
    with open('/home/tbrownex/data/Hackett/Valmont/colFrequency.csv', mode='r') as f:
        reader = csv.reader(f, delimiter='|')
        d = {rows[0]:rows[1] for rows in reader}
    return d