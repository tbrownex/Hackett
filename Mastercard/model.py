'''  '''

import modelSTL

__author__ = "The Hackett Group"

def process(train, args):
    if args.model == "stl":
        return modelSTL.process(train)