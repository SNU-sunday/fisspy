from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from interpolation.splines import LinearSpline, CubicSpline
# from scipy.interpolate import interp1d

__author__ = "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"
__all__ = ["ARcast"]

def ARcast(data, time, dt=False, axis=-1, missing=0):
    """
    Forecast the data by using AutoRegressive method.
    
    The code automatically find the unevenly sampled data point, 
    and then forecast the that point by using AR method.

    Parameters
    ----------
    data : `~numpy.ndarray`
        n dimensional data.
        Data must have the same number of elements to the time.
    time : astropy.time.core.Time
        The time for the each data points.
    dt : (optional) float
        An Interval of the time between each data in second unit.
    axis : (optional) int
        An axis to forecast.
    missing : (optional) float
        The missing value of the data.
        It may be due to data alignment.

    Returns
    -------
    ARdata : ~numpy.ndarray
        Autoregressived data.
        It must be larger elements then input data.
    tf : ~numpy.ndarray
        Time the forecasted ARdata points.
    
    Notes
    -----
    Input time must be the astropy.time.core.Time, 
    but output time is the ~numpy.ndarray.
    """
    None

# def ARcast_old(data,time,dt=False,axis=-1,missing=0):
#     """
#     Forecast the data by using AutoRegressive method.
    
#     The code automatically find the unevenly sampled data point, 
#     and then forecast the that point by using AR method.
    
#     Parameters
#     ----------
#     data : ~numpy.ndarray
#         n dimensional data.
#         Data must have the same number of elements to the time.
#     time : astropy.time.core.Time
#         The time for the each data points.
#     dt : (optional) float
#         An Interval of the time between each data in second unit.
#     axis : (optional) int
#         An axis to forecast.
#     missing : (optional) float
#         The missing value of the data.
#         It may be due to data alignment.
    
#     Returns
#     -------
#     ARdata : ~numpy.ndarray
#         Autoregressived data.
#         It must be larger elements then input data.
#     tf : ~numpy.ndarray
#         Time the forecasted ARdata points.
    
#     Notes
#     -----
#     Input time must be the astropy.time.core.Time, 
#     but output time is the ~numpy.ndarray.
    
    
#     Example
#     -------
#     >>> from fisspy.analysis.forecast import ARcast
#     >>> ARdata, tf = ARcast(data,t,dt=20.,axis=1)
#     """
#     if not dt:
#         dt=(time[1]-time[0]).value
    
#     shape=list(data.shape)
#     shape0=list(data.shape)
#     if shape[axis]!=len(time):
#         raise ValueError('The size of data is different from the size of time.')
        
#     t=(time-time[0])*24*3600
#     t=t.value
#     tf=np.arange(t[0],t[-1],dt,dtype=float)
    
#     interp=interp1d(t,data,axis=axis)
#     datai=interp(tf)
    
#     shape.pop(axis)
#     ind=[shape0.index(i) for i in shape]
#     ind=[axis]+ind
#     datat=datai.transpose(ind)
    
#     shapei=datat.shape
#     datat=datat.reshape((shapei[0],np.prod(shapei[1:])))
#     shapet=datat.shape
    
#     td=t-np.roll(t,1)
#     addi=np.where(td >= dt*2)[0]
    
#     for wh in addi:
#         for i in range(shapet[1]):
#             y=datat[:,i]
#             wh2=wh+int(td[wh]/dt-1)
#             if (y==missing).sum()<4:
#                 bar=AR(y)
#                 car=bar.fit()
#                 dar=car.predict(int(wh),int(wh2))
#                 datat[wh:wh2+1,i]=dar
#             else:
#                 datat[wh:wh2+1,i]=missing
#     datat=datat.reshape((shapei))
    
#     return datat.transpose(ind), tf
                