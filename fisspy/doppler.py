"""
Doppler


"""
from __future__ import absolute_import, division, print_function

__date__="Aug 08 2016"
__author__="J. Kang : jhkang@astro.snu.ac.kr"

import numpy as np
from scipy.interpolate import interp1d

def wavecalib(band,profile,method=True,pca=True):
    """
    
    """
    band=band[0:4]
    nw=profile.shape[0]
    
    if method:
        if band == '6562':
            line=np.array([6561.097,6564.206])
            lamb0=6562.817
            dldw=0.019182
        elif band == '8542':
            line=np.array([8540.817,8546.222])
            lamb0=8542.090
            dldw=-0.026252
        elif band == '5890':
            line=np.array([5889.951,5892.898])
            lamb0=5889.9509
            dldw=0.016847
        else:
            line=np.array([5434.524,5436.596])
            lamb0=5434.5235
            dldw=-0.016847
    else:
        if band == '6562':
            line=np.array([6562.817,6559.580])
            lamb0=6562.817
            dldw=0.019182
        else:
            line=np.array([8542.089,8537.930])
            lamb0=8542.090
            dldw=-0.026252
    
    
    w=np.arange(nw)
    wl=np.zeros(2)
    wc=profile[20:nw-20].argmin()+20
    lamb=(w-wc)*dldw+lamb0
    
    for i in range(2):
        mask=np.abs(lamb-line[i]) <= 0.3
        wtmp=w[mask]
        ptmp=np.convolve(profile[mask],[-1,2,-1],'same')
        mask2=ptmp[1:-1].argmin()+1
        try:
            wtmp=wtmp[mask2-3:mask2+4]
            ptmp=ptmp[mask2-3:mask2+4]
        except:
            raise ValueError('Fail to wavelength calibration\n'
            'please change the method %s to %s' %(repr(method), repr(not method)))
        c=np.polyfit(wtmp-np.median(wtmp),ptmp,2)
        wl[i]=np.median(wtmp)-c[1]/(2*c[0])    #local minimum of the profile
    
    dldw=(line[1]-line[0])/(wl[1]-wl[0])
    wc=wl[0]-(line[0]-lamb0)/dldw
    wavelength=(w-wc)*dldw
    
    return wavelength

def lambdameter(wv,data,hw):
    """
    FISS Doppler Lambdameter
    """
    
    s=data.argmin(axis=-1)
    sp0inter=interp1d(wv,data,axis=-1)
#    sp0=0.5*(sp0inter(wv[s]+hw)+sp0inter(wv[s]-hw))

