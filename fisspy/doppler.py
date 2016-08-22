"""
Doppler


"""
from __future__ import absolute_import, division, print_function

__date__="Aug 08 2016"
__author__="J. Kang : jhkang@astro.snu.ac.kr"

import numpy as np
#from scipy.interpolate import interp1d
from interpolation.splines import LinearSpline
import scipy

def wavecalib(band,profile,method=True):
    """
    FISS Wavecalibration
    
    Based on the IDL code written by (J. Chae 2013)
    
    Arguments
        band : a string wavelength band, '6562','8542','5890','5434'
        profile : a 1-D spectral profile
    
    Keywords
        Method : If true, the reference lines are the telluric lines.
                 else if False, the reference lines are the solar absorption lines.
    ==========================================
    Example)
    >>> wv=fisspy.doppler.wavecalib('6562',profile)
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


def lambdameter(wv,data,hw=0.,sp=5000.,wvinput=True):
    """
    FISS Doppler Lambdameter
    
    Determine the Lambdameter chord center for a given half width or intensity.
    
    Based on the IDL code written by (J. Chae)
    
    Arguments
        wv : A Calibrated wavelength.
        data : n-D (n>=2) spectral profile data, 
               the last dimension must be the spectral components
               and size is equal to wv.
               
        Case wvinput=True
        hw : A half width of the horizontal line segment
        outputs are an array of central wavelength values
                and an array of intensies of the line segment
        
        Case wvinput=False
        sp : A intensity of the horiznotal segment
        outputs are an array of central wavelength values
                and an array of half widths of the line segment
    
    =======================================================
    Example)
    >>> wc, inten=fisspy.doppler.labdameter(wv,data,0.2)
    """
    
    shape=data.shape
    nw=shape[-1]
    reshape=shape[:-1]
    if wv.shape[0] != nw:
        raise ValueError('The number of elements of wv and'
        'the number of elements of last axis for data are not equal.')
    
    na=int(data.size/nw)
    data=data.reshape((na,nw))
    s=data.argmin(axis=-1)

    
    if wvinput and hw == 0.:
        raise ValueError('The half-width value must be greater than 0.')
#        fna=range(na)
#        wtmp=wv[np.array((s-5,s-4,s-3,s-2,s-1,s,s+1,s+2,s+3,s+4,s+5))]
#        mwtmp=np.median(wtmp,axis=0)
#        sp0=np.array([data[i,s[i]-5:s[i]+6] for i in fna])
#        c=np.array([scipy.polyfit(wtmp[:,i]-mwtmp[i],sp0[i,:],2) for i in fna])
#        wc=mwtmp-c[:,1]/(2*c[:,0])
#        p=[scipy.poly1d(c[i,:]) for i in fna]
#        intc=np.array([p[i](wc[i]-mwtmp[i]) for i in fna])
#        wc=wc.reshape(reshape).T
#        intc=intc.reshape(reshape).T
#        return wc, intc
        
    posi0=np.arange(na)
    smin=[0,wv[0]]
    smax=[na-1,wv[-1]]
    order=[na,len(wv)]
    if wvinput:
            interp=LinearSpline(smin,smax,order,data)
            wl=np.array((posi0,wv[s]-hw)).T; wr=np.array((posi0,wv[s]+hw)).T
            intc=0.5*(interp(wl)+interp(wr))
    else:
        intc=np.ones(na)*sp
    
    wc=np.empty(na)
    hwc=np.empty(na)
    ref=1    
    rep=0
    more=np.ones(na,dtype=bool)
    
    while ref > 0.0001 and rep <6:
        sp1=data-intc[:,np.newaxis]
        comp=sp1[:,0:nw-1]*sp1[:,1:nw]
        
        s=comp[more] <=0.
        nsol=s.sum(axis=1)
        j=nsol//2
        whl=nsol.cumsum()-nsol+j-1
        whr=nsol.cumsum()-nsol+j
        whp, whs=np.where(s)
        l=whs[whl]
        r=whs[whr]
        posi=posi0[more]
        wl0=wv[l]-(wv[l+1]-wv[l])/(sp1[posi,l+1]-sp1[posi,l])*sp1[posi,l]
        wr0=wv[r]-(wv[r+1]-wv[r])/(sp1[posi,r+1]-sp1[posi,r])*sp1[posi,r]
        wc[more]=0.5*(wl0+wr0)
        hwc[more]=0.5*np.abs(wr0-wl0)
        
        if wvinput:
            wl=np.array((posi,wc[more]-hw)).T; wr=np.array((posi,wc[more]+hw)).T
            intc[more]=0.5*(interp(wl)+interp(wr))
            ref0=np.abs(hwc-hw)
            ref=ref0.max()
            more=ref0>0.0001
        else:
            ref=0
        rep+=1
    
    wc=wc.reshape(reshape).T
    if wvinput:
        intc=intc.reshape(reshape).T
        return wc, intc
    else:
        hwc=hwc.reshape(reshape).T
        return wc, hwc