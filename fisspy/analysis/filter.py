from __future__ import absolute_import, division
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

def FourierFilter(data, nt, dt, filterRange, axis=0):
    """
    """
    freq = fftfreq(nt, dt)
    if filterRange[0] == None:
        filterRange[0] = 0
    if filterRange[1] == None:
        filterRange[1] = freq.max()
    filt = np.logical_or(np.abs(freq) < filterRange[0],
                         np.abs(freq) > filterRange[1])
    fdata = fft(data, axis=axis)
    fdata[filt] = 0
    return ifft(fdata, axis=axis).real