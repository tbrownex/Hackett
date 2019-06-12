''' An abstract report writer object with methods to:
    - create an output file
    - add records to the file
    - close the file.
    
    The name and location of the file are input parameters'''

__author__ = "The Hackett Group"

import csv

class Report:
    
    def __init__(self, loc, name):
        self.csvfile = open(loc + name, 'w')
        self.csvwriter = csv.writer(self.csvfile, delimiter=",")
    
    def writeRow(self, rec):
        self.csvwriter.writerow(rec)
        
    def close(self):           
        self.csvfile.close()