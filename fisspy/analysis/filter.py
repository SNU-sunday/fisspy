from __future__ import absolute_import, division
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

__all__ = ["FourierFilter"]

def FourierFilter(data, nt, dt, filterRange, axis=0):
    """
    Apply the Fourier bandpass filter.

    Parameters
    ----------
    data: `~numpy.ndarray`
        N-dimensional array.
    nt: `int`
        The number of the time domain.
    dt: `float`
        Bin size of the time domain.
    filterRange: `list`
        Bandpass filter range.
    axis: `int`
        time axis of the data.
    """
    if data.dtype == '>f8':
        data = data.astype(float)
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
