'''  '''

import forecastSTL

__author__ = "The Hackett Group"

def process(model, config, args):
    if args.model == "stl":
        return forecastSTL.process(model, config)