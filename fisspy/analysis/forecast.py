from statsmodels.tsa.ar_model import AR
import numpy as np
from scipy.interpolate import interp1d

def ARcast(data,time,dt=False,axis=-1):
    """
    """
    if not dt:
        dt=(time[1]-time[0]).value
    
    shape=list(data.shape)
    shape0=list(data.shape)
    if shape[axis]!=len(time):
        raise ValueError('The size of data is different from the size of time.')
        
    t=(time-time[0])*24*3600
    t=t.value
    tf=np.arange(t[0],t[-1],dt,dtype=float)
    
    interp=interp1d(t,data,axis=axis)
    datai=interp(tf)
    
    shape.pop(axis)
    ind=[shape0.index(i) for i in shape]
    ind=[axis]+ind
    datat=datai.transpose(ind)
    
    shapei=datat.shape
    datat=datat.reshape((shapei[0],np.prod(shapei[1:])))
    shapet=datat.shape
    
    td=t-np.roll(t,1)
    addi=np.where(td >= dt*2)[0]
    
    for wh in addi:
        for i in range(shapet[1]):
            bar=AR(datat[:,i])
            car=bar.fit()
            wh2=wh+int(td[wh]/dt-1)
            dar=car.predict(int(wh),int(wh2))
            datat[wh:wh2+1,i]=dar
        
    datat=datat.reshape((shapei))
    
    return datat.transpose(ind), tf
                