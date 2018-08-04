# Use the Dickey Fuller test for stationarity
# CI is the confidence interval
# prt is the binary print flag

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def powerDensity(series, freq, store, plot):
    #f, Pxx_den = signal.periodogram(series, fs=freq, detrend='linear')
    f, density = signal.periodogram(series, fs=freq, detrend='linear')
    normed = density / density.sum()
    if plot:
        plt.semilogy(f[:60], normed[:60])
        plt.ylim([1e-7, 1e2])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title(store)
        plt.show()
    
    if np.sqrt(normed.max()) > .4:
        seasonal = True
        freq = f[np.argmax(normed)]
    else:
        seasonal = False
    return seasonal, freq