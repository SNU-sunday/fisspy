from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special._ufuncs import gamma, gammainc
from scipy.optimize import fminbound as fmin
from scipy.fftpack import fft, ifft

__author__ = "J. Kang : jhkang@astro.snu.ac.kr"
__date__= "Sep 13 2016"

def wavelet(y, dt,
            dj=0.25, mother='MORLET',
            s0=False, j=False, param=False, pad=False):
    
    n=len(y)
    n0=n
    if not s0:
        s0 = 2*dt
    if not j:
        j = int(np.log2(n*dt/s0)/dj)
    else:
        j=int(j)
    #reconstruct the time series to analyze if set pad
    x = y - y.mean()
    if pad:
        power = int(np.log2(n)+0.4999)
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
    
    nowf, period, fourier_factor, coi = motherfunc(mother,
                                                           k, scale,param)
    wave = ifft(fx*nowf)
    coi=coi*dt*np.append(np.arange((n0+1)//2),np.arange(n0//2-1,-1,-1))
    
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
        raise ValueError('Mother must be one of MORLET, PAUL, DOG\n'
                         'mother = %s' %repr(mother))
    period = scale2*fourier_factor
    return nowf, period, fourier_factor, coi

def wave_signif(y,dt,scale,sigtest=0,mother='MORLET',
                param=False,lag1=0.0,siglvl=0.95,dof=-1,
                gws=False,confidence=False):
    
    if len(y) == 1:
        var = y
    else:
        var = np.var(y)
    
    j = len(scale)
    
    if mother == 'MORLET':
        if not param:
            param = 6.
        fourier_factor = 4*np.pi/(param+(2+param**2)**0.5)
        dofmin=2.
        if param == 6.:
            cdelta = 0.776
            gamma_fac = 2.32
            dj0 = 0.60
        else:
            cdelta = -1
            gamma_fac = -1
            dj0 = -1
    elif mother == 'PAUL':
        if not param:
            param = 4.
        fourier_factor = 4*np.pi/(2*param+1)
        dofmin = 2.
        if param == 4.:
            cdelta = 1.132
            gamma_fac = 1.17
            dj0 = 1.5
        else:
            cdelta = -1
            gamma_fac = -1
            dj0 = -1
    elif mother == 'DOG':
        if not param:
            param = 2.
        fourier_factor = 2.*np.pi*(2./(2*param+1))**0.5
        dofmin = 1.
        if param == 2.:
            cdelta = 3.541
            gamma_fac = 1.43
            dj0 = 1.4
        elif param ==6.:
            cdelta = 1.966
            gamma_fac = 1.37
            dj0 = 0.97
        else:
            cdelta = -1
            gamma_fac = -1
            dj0 = -1
    else:
        raise ValueError('Mother must be one of MORLET, PAUL, DOG')
    
    period = scale*fourier_factor
    freq = dj0/period
    fft_theor = (1-lag1**2)/(1-2*lag1*np.cos(freq*2*np.pi)+lag1**2)
    fft_theor*=var
    if gws:
        fft_theor = gws
    signif = fft_theor
    
    if sigtest == 0:
        dof = dofmin
        signif = fft_theor * chisquare_inv(siglvl, dof)/dof
        if confidence:
            sig = (1.-siglvl)/2.
            chisqr = dof/np.array((chisquare_inv(1-sig,dof),
                                   chisquare_inv(sig,dof)))
            signif = np.dot(chisqr[:,np.newaxis],fft_theor[np.newaxis,:])
    elif sigtest == 1:
        if gamma_fac == -1:
            raise ValueError('gamma_fac(decorrelation facotr) not defined for '
                             'mother = %s with param = %s'
                             %(repr(mother),repr(param)))
        if dof == -1:
            dof = dofmin
        if len(np.atleast_1d(dof)) == 1:
            dof = np.zeros(j)+dof
        dof[dof <= 1] = 1
        dof = dofmin*(1+(dof*dt/gamma_fac/scale)**2)**0.5
        dof[dof <= dofmin] = dofmin
        if not confidence:
            for i in range(j):
                chisqr = chisquare_inv(siglvl,dof[i])/dof[i]
                signif[i] = chisqr*fft_theor[i]
        else:
            signif = np.empty(2,j)
            sig = (1-siglvl)/2.
            for i in range(j):
                chisqr = dof[i]/np.array((chisquare_inv(1-sig,dof[i]),chisquare_inv(sig,dof[i])))
                signif[:,i] = fft_theor[i]*chisqr
    elif sigtest == 2:
        if len(dof) != 2:
            raise ValueError('DOF must be set to [s1,s2], the range of scale-averages')
        if cdelta != -1:
            raise ValueError('cdelta & dj0 not defined for'
                             'mother = %s with param = %s' %(repr(mother),repr(param)))
        dj= np.log2(scale[1]/scale[0])
        s1 = dof[0]
        s2 = dof[1]
        avg = (scale>=s1)*(scale<=s2)
        navg = avg.sum()
        if not navg:
            raise ValueError('No valid scales between %s and %s' %(repr(s1),repr(s2)))
        savg = 1./(1./scale[avg]).sum()
        smid = np.exp(0.5*np.log(s1*s2))
        dof = (dofmin*navg*savg/smid)*(1+(navg*dj/dj0)**2)**0.5
        fft_theor = savg*(fft_theor[avg]/scale[avg]).sum()
        chisqr = chisquare_inv(siglvl,dof)/dof
        if confidence:
            sig = (1-siglvl)/2.
            chisqr = dof/np.array((chisquare_inv(1-sig,dof),chisquare_inv(sig,dof)))
        signif = (dj*dt/cdelta/savg)*fft_theor*chisqr
    else:
        raise ValueError('Sigtest must be 0,1, or 2')
    return signif
def chisquare_inv(p,v):
    """
    CHISQUARE_INV
    
    Inverse of chi-square cumulative distribution function(CDF).
    
    Return the inverse of chi-square cdf
    
    parameter
    p : probability
    v : degrees of freedom of the chi-square distribution
    =====================================
    Example
    >>> result = chisquare_inv(p,v)
    """
    if not 0<p<1:
        raise ValueError('p must be 0<p<1')
    minv = 0.01
    maxv = 1
    x = 1
    tolerance = 1e-4
    while x+tolerance >= maxv:
        maxv*=10.
        x = fmin(chisquare_solve, minv, maxv, args=(p,v), xtol=tolerance)
        minv = maxv
    x*=v
    return x
    
def chisquare_solve(xguess,p,v):
    pguess = gammainc(v/2,v*xguess/2)
    pdiff = np.abs(pguess - p)
    if pguess >= 1-1e-4:
        pdiff = xguess
    return pdiff
