import numpy as np
from scipy.signal.windows import tukey

__author__ = "Juhyung Kang"
__all__ = ['detrend', 'pfcurve']

def detrend(y, order=3, alpha=0.05):
    nd = len(y)
    x = np.arange(nd)
    fy = pfcurve(x,y,order)
    w = tukey(nd, alpha=alpha)
    return (y-fy)*w

def pfcurve(x, y, order):
    par = np.polyfit(x, y, order)
    fy = np.polyval(par, x)
    return fy