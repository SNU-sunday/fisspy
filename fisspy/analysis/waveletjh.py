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
    n0=n.copy
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
    
    nowf, period, fourier_factor, coi, dofmin = motherfunc(mother,
                                                           k, scale,param)
    wave = ifft(fx*nowf)
    coi=coi*dt*np.append(np.arange((n0+1)//2),np.arange((n0//2-1,-1,-1)))
    
    return wave[:,:n0], period, scale, coi

def motherfunc(mother, k, scale, param):
    """"""
    n = len(k)
    kp = k > 0.
    scale2 = scale[:,np.newaxis]
    pi = np.pi
    
    if mother == 'MORLET':
        if not param:
            param = 6.
        expn = -(scale2*k-param)**2/2.*kp
        norm = pi**-0.25*(n*k[1]*scale2)**0.5
        nowf = norm*np.exp(expn)*kp*(expn > -100.)
        fourier_factor = 4*pi/(param+(2+param**2)**0.5)
        coi = fourier_factor/2**0.5
        
    elif mother == 'PAUL':
        if not param:
            param = 4.
        expn = -scale2*k*kp
        norm = 2**param*(scale2*k[1]*n/(param*gamma(2*param)))**0.5
        nowf = norm*np.exp(expn)*((scale2*k)**param)*kp*(expn > -100.)
        fourier_factor = 4*pi/(2*param+1)
        coi = fourier_factor*2**0.5
        
    elif mother == 'DOG':
        if not param:
            param = 2.
        expn = -(scale2*k)**2/2.
        norm = (scale2*k[1]*n/gamma(param+0.5))**0.5
        nowf = -norm*1j**param*(scale2*k)**param*np.exp(expn)
        fourier_factor = 2*pi*(2./(2*param+1))**0.5
        coi = fourier_factor/2**0.5
    else:
        raise ValueError('mother must be one of MORLET, PAUL, DOG')
    period = scale2*fourier_factor
    return nowf, period, fourier_factor, coi

def wavesignif(y,dt,scale):
    