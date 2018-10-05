''' Set the logging level for the process'''

__author__ = "The Hackett Group"

import logging

def setLogging(config):
    # Validate the logging level
    logLevel = getattr(logging, config["logDefault"].upper(), None)
    if not isinstance(logLevel, int):
        raise ValueError('Invalid log level: %s' % args.log)

    logging.basicConfig(filename=config["logLoc"]+config["logFile"],\
                        level=logLevel,\
                        format='%(levelname)s:%(message)s')