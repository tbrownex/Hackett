# timestampGenerator 
# -*- coding: utf-8 -*-


"""Timestamp Generator.
The job of the timestamp generator is to create a epoch from the current time
of day.
The function takes zero arguments and will return a string formatted epoch 
(Number of seconds since Jan. 1 1970).

ex:
tstamp()

returns:
'1536852781'
"""

import time

def tstamp():
    tstamp = str(int(time.time()))
    return tstamp