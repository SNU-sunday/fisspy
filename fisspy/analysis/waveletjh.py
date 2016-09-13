from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special._ufuncs import gamma, gammainc
from scipy.optimize import fminbound
from scipy.fftpack import fft, ifft

__author__ = "J. Kang : jhkang@astro.snu.ac.kr"
__date__= "Sep 13 2016"

def wavelet(y, dt,
            dj=0.25, mother='MORLET',
            s0=False, j=False, param=False, pad=False):
    
    n=len(y)
    if not s0:
        s0 = 2*dt
    if not j:
        j = int(np.log(n*dt/s0)/np.log(2)/dj)
    
    #reconstruct the time series to analyze if set pad
    x = y - y.mean()
    if pad:
        power = int(np.log(n)/np.log(2)+0.4999)
        x = np.append(x,np.zeros(2**(power+1)-n))
        n=len(x)
    
    #wavenumber
    k1 = np.arange(1,n//2+1)*2.*np.pi/n/dt
    k2 = -k1[:int((n-1)/2)][::-1]
    k = np.concatenate(([0.],k1,k2))
    
    #Scale array
    scale=s0*2.**(np.arange(j+1,dtype=float)*dj)
    
    # FFT
    fx = fft(x)
    
    
    return wave

def motherfunc(mother, k, scale, param):
    n = len(k)
    k2 = k > 0.
    scale2=scale[:,np.newaxis]
    
    if mother == 'MORLET':
        if not param:
            param = 6.
        
        expn=-(scale2*k-param)**2/2.*k2
        undermask=expn > -100.
        norm=np.pi**-0.25*(n*k[1]*scale2)**0.5
        psi_fft=norm*np.exp(expn)*k2*undermask
        
    elif mother == 'PAUL':
        
    elif mother == 'DOG':
        
    else:
        raise ValueError('mother must be one of MORLET, PAUL, DOG')
    