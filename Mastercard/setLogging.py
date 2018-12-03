__author__ = "The Hackett Group"

import logging

def setLogging(config, args):
    '''
    Set the logging level for the process
    '''

    # Validate the logging level
    if args.log:
        logLevel = getattr(logging, args.log.upper(), None)
    else:
        logLevel = getattr(logging, config["logDefault"].upper(), None)
    if not isinstance(logLevel, int):
        raise ValueError('Invalid log level: %s' % args.log)

    logging.basicConfig(filename=config["logLoc"]+config["logFile"],\
                        level=logLevel,\
                        format='%(levelname)s:%(message)s')