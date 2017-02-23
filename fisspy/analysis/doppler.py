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

__all__ = ['wavecalib', 'lambdameter','LOS_velocity']

def wavecalib(band,profile,method=True):
    """
    Calibrate the wavelength for FISS spectrum profile.
    
    Parameters
    ----------
    band : str
        A string to identify the wavelength.
        Allowable wavelength bands are '6562','8542','5890','5434'
    profile : ~numpy.ndarray
        A 1 dimensional numpy array of spectral profile.
    Method : (optional) bool
        * Default is True.
        If true, the reference lines for calibration are the telluric lines.
        Else if False, the reference lines are the solar absorption lines.
    
    Returns
    -------
    wavelength : ~numpy.ndarray
        Calibrated wavelength.
    
    Notes
    -----
        This function is based on the FISS IDL code FISS_WV_CALIB.PRO
        written by J. Chae, 2013.
    
    Example
    -------
    >>> from fisspy.analysis import doppler
    >>> wv=doppler.wavecalib('6562',profile)
    
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
        elif band == '5434':
            line=np.array([5434.524,5436.596])
            lamb0=5434.5235
            dldw=-0.016847
        else:
            raise ValueError("The wavelength band value is not allowable.\n"+
                             "Please select the wavelenth "+
                             "among '6562','8542','5890','5434'")
    else:
        if band == '6562':
            line=np.array([6562.817,6559.580])
            lamb0=6562.817
            dldw=0.019182
        elif band == '8542':
            line=np.array([8542.089,8537.930])
            lamb0=8542.090
            dldw=-0.026252
        else:
            raise ValueError("The wavelength band value is not allowable.\n"
                             "Please select the wavelenth "
                             "among '6562','8542','5890','5434'")
    
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


def lambdameter(wv,data0,hw=0.03,sp=5000,wvinput=True):
    """
    Determine the Lambdameter chord center for a given half width or intensity.
    
    Parameters
    ----------
    wv : ~numpy.ndarray
        A Calibrated wavelength.
    data : ~numpy.ndarray
        n (n>=2) dimensional spectral profile data, 
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
    if wv.shape[0] != nw:
        raise ValueError('The number of elements of wv and '
        'the number of elements of last axis for data are not equal.')
    
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
        wl0=wv[l]-(wv[l+1]-wv[l])/(sp1[posi,l+1]-sp1[posi,l])*sp1[posi,l]
        wr0=wv[r]-(wv[r+1]-wv[r])/(sp1[posi,r+1]-sp1[posi,r])*sp1[posi,r]
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
    
    wc=wc.reshape(reshape)
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
    if not band:
        raise ValueError("Please insert the parameter band (str)")
        
    wc, intc =  lambdameter(wv,data,hw,wvinput=True)
    
    if band=='6562':
        return wc*c.to('km/s').value/6562.817
    elif band=='8542':
        return wc*c.to('km/s').value/8542.09
    elif band=='5890':
        return wc*c.to('km/s').value/5890.9399
    elif band=='5434':
        return wc*c.to('km/s').value/5434.3398
    else:
        raise ValueError("Value of band must be one among"
                         "'6562', '8542', '5890', '5434'")