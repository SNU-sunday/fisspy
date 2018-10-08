"""
Doppler

This module calculate line of sight doppler velocities for 
each pixels of a FISS fts data.
"""
from __future__ import absolute_import, division

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"

import numpy as np
from interpolation.splines import LinearSpline
from astropy.constants import c
from scipy.signal import fftconvolve as conv
from fisspy.image.base import alignoffset

__all__ = ['lambdameter', 'LOS_velocity']




def lambdameter(wv, data0, ref_spectrum= False, wvRange = False,
                hw= 0.03, sp= 5000, wvinput= True):
    """
    Determine the Lambdameter chord center for a given half width or intensity.
    
    Parameters
    ----------
    wv : ~numpy.ndarray
        A Calibrated wavelength.
    data : ~numpy.ndarray
        n (n=2 or n=3) dimensional spectral profile data, 
        the last dimension component must be the spectral component,
        and the size is equal to the size of wv.
    wvinput : bool
        There are two cases.
            
    * Case wvinput==True
        
            hw : float
                A half width of the horizontal line segment.
                
        Returns
        -------
        wc : nd ndarray
            n dimensional array of central wavelength values.
        intc : nd ndarray
            n dimensional array of intensies of the line segment.\\
        
    * Case wvinput==False
        
            sp : float
                An intensity of the horiznotal segment.
                
        Returns
        -------
        wc : nd ndarray
            n dimensional array of central wavelength values.
        hwc : nd ndarray
            n dimensional array of half widths of the line segment.
    
    Notes
    -----
        This function is based on the IDL code BISECTOR_D.PRO
        written by J. Chae.
    
    Example
    -------
    >>> from fisspy.analysis import doppler
    >>> wc, inten = doppler.labdameter(wv,data,0.2)
    
    """
    
    shape=data0.shape
    nw=shape[-1]
    reshape=shape[:-1]
    dkern = np.array([[-1, 1, 0, 1, -1]])
    rspec = np.any(ref_spectrum)
    ndim = data0.ndim
    wvoffset = 0
    dwv = wv[1]-wv[0]
    if rspec and data0.ndim == 3:
        refSpec = conv(ref_spectrum , dkern[0],'same')
        refSpec[:2] = refSpec[-2:] = 0
        refSpec = refSpec * np.ones((4,1))
        data2d = conv(data0.mean(0), dkern, 'same')
        data2d[:,:2] = data2d[:,-2:] = 0
        data = data2d * np.ones((4, 1, 1))
#        data[:,:,:2] = data[:,:,-2:] = 0
        dataT = data.transpose((1, 0, 2))
        yoff, xoff, cor = alignoffset(dataT, refSpec, cor= True)
        wvoffset = (xoff*(wv[1]-wv[0])) * (cor > 0.7)
    elif not rspec and ndim == 3:
        wvoffset = np.zeros(shape[1])
    elif ndim == 1 or ndim >=4:
        ValueError('The dimension of data0 must be 2 or 3.')
    
    if wv.shape[0] != nw:
        raise ValueError('The number of elements of wv and '
        'the number of elements of last axis for data are not equal.')
    
    if np.any(wvRange):
        ss = np.logical_and(wv >= wvRange[0], wv <= wvRange[1])
        nw = ss.sum()
        data0 = data0[:,:,ss].copy()
        wv = wv[ss].copy()
    na=int(data0.size/nw)
    data=data0.reshape((na,nw))

    
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
    
    wc=np.zeros(na)
    hwc=np.zeros(na)
    ref=1    
    rep=0
    s0=s.copy()
    more=data[posi0,s0]>100
    
    while ref > 0.00001 and rep <6:
        sp1=data-intc[:,None]
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
        wl0=wv[l]-dwv/(sp1[posi,l+1]-sp1[posi,l])*sp1[posi,l]
        wr0=wv[r]-dwv/(sp1[posi,r+1]-sp1[posi,r])*sp1[posi,r]
        wc[more]=0.5*(wl0+wr0)
        hwc[more]=0.5*np.abs(wr0-wl0)
        
        if wvinput:
            wl=np.array((posi,wc[more]-hw)).T; wr=np.array((posi,wc[more]+hw)).T
            intc[more]=0.5*(interp(wl)+interp(wr))
            ref0=np.abs(hwc-hw)
            ref=ref0.max()
            more=(ref0>0.00001)*(data[posi0,s0]>100)
        else:
            ref=0
        rep+=1
    

    wc = wc.reshape(reshape) - wvoffset
    if wvinput:
        intc=intc.reshape(reshape)
        return wc, intc
    else:
        hwc=hwc.reshape(reshape)
        return wc, hwc

def LOS_velocity(wv,data,hw=0.01,band=False):
    """
    Calculte the Line-of-Sight velocity of given data.
    
    Parameters
    ----------
    wv : ~numpy.ndarray
        A Calibrated wavelength.
    data : ~numpy.ndarray
        n (n>=2) dimensional spectral profile data, 
        the last dimension component must be the spectral component,
        and the size is equal to the size of wv.
    hw : float
        A half width of the horizontal line segment.
    band : str
        A string of the wavelength band.
        It must be the 4 characters in Angstrom unit. ex) '6562', '8542'
        
    Returns
    -------
    losv : ~numpy.ndarray
        n-1 (n>=2) dimensional Line-of_sight velocity value, where n is the
        dimension of the given data.
        
    Example
    -------
    >>> from fisspy.doppler import LOS_velocity
    >>> mask = np.abs(wv) < 1
    >>> losv = LOS_velocity(wv[mask],data[:,:,mask],hw=0.03,band='6562')
    """
    if not band :
        raise ValueError("Please insert the parameter band (str)")
        
    wc, intc =  lambdameter(wv,data,hw,wvinput=True)
    
    if band == '6562' :
        return wc*c.to('km/s').value/6562.817
    elif band == '8542' :
        return wc*c.to('km/s').value/8542.09
    elif band == '5890' :
        return wc*c.to('km/s').value/5890.9399
    elif band == '5434' :
        return wc*c.to('km/s').value/5434.3398
    else:
        raise ValueError("Value of band must be one among"
                         "'6562', '8542', '5890', '5434'")
        
