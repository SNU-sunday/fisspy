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
    FISS Wavecalibration
    
    Based on the IDL code written by (J. Chae 2013)
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
    """
    
    shape=data.shape
    nw=shape[-1]
    reshape=shape[:-1]
    if wv.shape[0] != nw:
        raise ValueError('The dimensions of %s and %s are not equal.'%(repr(wv),repr(data)))
    
    na=int(data.size/nw)
    data=data.reshape((na,nw))
    
    s=data.argmin(axis=-1)
    wc=np.zeros(na)
    intc=np.zeros(na)
    
    if wvinput and hw == 0.:
        wtmp=wv[np.array((s-5,s-4,s-3,s-2,s-1,s,s+1,s+2,s+3,s+4,s+5))]
        mwtmp=np.median(wtmp,axis=0)
        for i in range(na):
            sp0=data[i,s[i]-5:s[i]+6]
            c=np.polyfit(wtmp[:,i]-mwtmp[i],sp0,2)
            wc[i]=mwtmp[i]-c[1]/(2*c[0])
            p=np.poly1d(c)
            intc[i]=p(wc[i]-mwtmp[i])
        wc=wc.reshape(reshape).T
        intc=intc.reshape(reshape).T
        return wc, intc
    
    if wvinput:
        interp=[None]*na
        for i in range(na):
            interp[i]=interp1d(wv,data[i,:])
            intc[i]=0.5*(interp[i](wv[s[i]]-hw)+interp[i](wv[s[i]]+hw))
    else:
        intc=np.ones(na)*sp
    
    hwc=np.zeros(na)
    ref=1    
    rep=0
    
    while ref > 0.001 or rep <5:
        sp1=data-intc[:,np.newaxis]*np.ones(nw)
        comp=sp1[:,0:nw-1]*sp1[:,1:nw]
    
        for i in range(na):
            s=np.where(comp[i,:] <= 0.)[0]
            nsol=s.size
            j=int(nsol/2)
            l=s[j-1]
            r=s[j]
            wl=wv[l]-(wv[l+1]-wv[l])/(sp1[i,l+1]-sp1[i,l])*sp1[i,l]
            wr=wv[r]-(wv[r+1]-wv[r])/(sp1[i,r+1]-sp1[i,r])*sp1[i,r]
            wc[i]=0.5*(wl+wr)
            hwc[i]=0.5*np.abs(wr-wl)
            
            if wvinput:
                intc[i]=0.5*(interp[i](wc[i]-hw)+interp[i](wc[i]+hw))
        if wvinput:
            ref=np.abs(hwc-hw).max()
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
