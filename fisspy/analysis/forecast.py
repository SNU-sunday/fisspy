from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from interpolation.splines import LinearSpline, CubicSpline
from scipy.interpolate import interp1d

__author__ = "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"
__all__ = ["ARcast"]

def ARcast(data, time, dt=False):
    """
    Forecast the data by using AutoRegressive method.
    
    The code automatically find the unevenly sampled data point, 
    and then forecast the that point by using AR method.

    Parameters
    ----------
    data : `~numpy.ndarray`
        n dimensional data.
        Data must have the same number of elements to the time.
        Data should be (nt,...) where nt is the size in the time axis.
    time : `~numpy.ndarray`
        The time for the each data points.
    dt : `float` (optional)
        An Interval of the time between each data in second unit.

    Returns
    -------
    ARdata : `~numpy.ndarray`
        Autoregressived data.
        It must be larger elements then input data.
    tf : `~numpy.ndarray`
        Time the forecasted ARdata points.
    
    """
    if not dt:
        tmp = np.roll(time, -1) - time
        DT = np.median(tmp[:-1])
    
    shape = data.shape
    if shape[0]!=len(time):
        raise ValueError('The size of data is different from the size of time.')
        
    t = (time-time[0])
    tf = np.arange(t[0], t[-1], DT, dtype=float)
    
    interp = interp1d(t, data, kind='cubic', axis=0)
    datai = interp(tf)
    shapei = datai.shape
    nt = len(tf)
    
    if data.ndim == 1:
        nl = 1
    else:
        nl = int(datai.size//nt)
    datat = datai.reshape((nt, nl))

    # shapei=datat.shape
    # datat=datat.reshape((shapei[0],np.prod(shapei[1:])))
    # shapet=datat.shape
    
    td = t - np.roll(t,1)
    addi = np.where(td >= DT*1.5)[0]

    nad = len(addi)
    if nad == 0:
        print("There is no empty element. Just interpolate to evenly sample the data.")
        return datai

    npredict = (td[addi]/DT + 0.5).astype(int)-1
    
    for i in range(nad):
        ts = np.abs(tf - t[addi[i]-1]).argmin()+1
        te = ts+npredict[i]
        
        if ts//2 < 10:
            fskip = True
        else:
            fskip = False
        if (nt - te)//2 < 10:
            bskip = True
        else:
            bskip = False

        bF0 = (np.arange(npredict[i])+1)/(npredict[i]+1)
        fF0 = 1 - bF0
        bF = (1-bskip)*(bF0 + fskip*fF0)
        fF = (1-fskip)*(fF0 + bskip*bF0)
        
        for l in range(nl):
            y = datat[:,l]
            if not fskip:
                fm = AutoReg(y[:ts], lags=10).fit()
                fp = fm.forecast(npredict[i])
            else:
                fp = np.zeros(npredict[i])
            if not bskip:
                bm = AutoReg(y[te:][::-1], lags=10).fit()
                bp = bm.forecast(npredict[i])
            pred = fF*fp + bF*bp[::-1]
            datat[ts:te,l] = pred

    datat = datat.reshape((shapei))
    
    return datat, tf+time[0]