"""
Wavelet
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special._ufuncs import gamma, gammainc
from scipy.optimize import fminbound as fmin
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.ticker import sf

__author__ = "J. Kang : jhkang@astro.snu.ac.kr"
__date__= "Sep 13 2016"

def wavelet(y, dt,
            dj=0.25, s0=False, j=False,
            mother='MORLET', param=False, pad=False):
    """
    Wavelet
    Computes the wavelet transform of the vecotr y
    with sampling rate dt.
    
    By default, the MORLET wavelet (k0=6) is used.
    The wavelet basis is normalized to have total energy=1
    at all scales.
    
    Parameters
        y      : The time series of length N.
        dt     : The time step between each y values
                 i.e. the sampling time.
        dj     : The spacing between discrete scales.(optional)
                 Default is 0.25
                 The smaller, the better scale resolution.
        s0     : The smallest scale of the wavelet.(optional)
                 Default is 2*dt.
        j      : The number of scales minus one.(optional)
                 Scales range from s0 up to s0*2**(j*dj), to give
                 a total of (j+1) scales.
                 Default is j=log2(n*dt/s0)/dj.
        mother : The mother wavelet function.(optional)
                 The choices are 'MORLET', 'PAUL', or 'DOG'
                 Default is 'MORLET'
        param  : The mother wavelet parameter.(optional)
                 For 'MORLET' param is k0, default is 6.
                 For 'PAUL' param is m, default is 4.
                 For 'DOG' param is m, default is 2.
    
    Keywords
        pad : If set True, pad time series with enough zeros to get
              N up to the next higher power of 2.
              This prevents wraparound from the end of the time series
              to the beginning, and also speeds up the FFT's 
              used to do the wavelet transform.
              This will not eliminate all edge effects.
    
    Outputs
        wave   : The WAVELET transform of y.
                 (j+1,n) complex arry.
                 np.arctan2(wave.imag,wave.real) gives the WAVELET phase.
                 wave.real gives the WAVELET amplitude.
                 The WAVELET power spectrum is np.abs(wave)**2
        period : The vecotr of "Fourier" periods (in time units)
                 that correspods to the scales.
        scale  : The vecotr of scale indices, given by s0*2**(j*dj),
                 j=0...j
                 where j+1 is the total number of scales.
        coi    : The Cone-of-Influence, which is a vector of N points
                 that contains the maximum period of useful information
                 at that particular time.
                 Periods greater than this are subject to edge effets.
    
    """
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

def iwavelet(wave,scale,dt,dj=0.25,mother='MORLET',param=False):
    a, b = wave.shape
    c = len(scale)
    scale2=1/scale**0.5
    
    if a != c:
        raise ValueError('Input array dimensions do mot match.')
    
    fourier_factor, dofmin, cdelta, gamma_fac, dj0 = motherparam(mother,param)
    if cdelta == -1:
        raise ValueError('Cdelta undefined, cannot inverse with this wavelet')
    
    if mother == 'MORLET':
        psi0=np.pi**(-0.25)
    elif mother == 'PAUL':
        psi0=2**param*gamma(param+1)/(np.pi*gamma(2*param+1))**0.5
    elif mother == 'DOG':
        if not param:
            param=2
        if param==2:
            psi0=0.867325
        elif param==6:
            psi0=0.88406
    
    iwave=dj*dt**0.5/(cdelta*psi0)*np.dot(scale2,wave.real)
    return iwave

def motherfunc(mother, k, scale, param):
    """
    Motherfunc
    
    computes the wavelet function as a function of Fourier frequency,
    used for the wavelet transform in Fourier space.
    
    Parameters
        mother  : A string, Equal to 'MORLET' or 'PAUL' or 'DOG'
        k       : The Fourier frequencies at which to calculate the wavelet
        scale   : The wavelet scale
        param   : The nondimensional parameter for the wavelet function
    
        Outputs
        nowf   : The nonorthogonal wavelet function
        fourier_factor : the ratio of Fourier period to scale
        coi    : The cone-of-influence size at the scale
        dofmin : Degrees of freedom for each point in the wavelet power
                 (either 2 for MORLET and PAUL, or 1 for the DOG)
    """
    n = len(k)
    kp = k > 0.
    scale2 = scale[:,np.newaxis]
    pi = np.pi
    
    if mother == 'MORLET':
        if not param:
            param = 6.
        expn = -(scale2*k-param)**2/2.*kp
        norm = pi**(-0.25)*(n*k[1]*scale2)**0.5
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
    period = scale*fourier_factor
    return nowf, period, fourier_factor, coi

def motherparam(mother,param=False):
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
    return fourier_factor, dofmin, cdelta, gamma_fac, dj0
    
def wave_signif(y,dt,scale,sigtest=0,mother='MORLET',
                param=False,lag1=0.0,siglvl=0.95,dof=-1,
                gws=False,confidence=False):
    """
    Wave_signif
    
    Compute the significance levels for a wavelet transform.
    
    Parameters
        y     : The time series, or the variance of the time series.
                If this is a single number, it is assumed to be the variance
        dt    : The sampling time.
        scale : The vecotr of scale indices, from previous call to WAVELET
        sigtest : 0, 1, or 2 (optional)
                  if 0 (default), then just do a regular chi-square test
                      i.e. Eqn (18) from Torrence & Compo.
                  If 1, then do a "time-average" test, i.e. Eqn (23).
                      in this case, dof should be set to False,
                      the nuber of local wavelet spectra 
                      that were averaged together.
                      For the Global Wavelet Spectrum(GWS), this would be N,
                      where N is the number of points in y
                  If 2, then do a "scale-average" test, i.e. Eqns (25)-(28).
                      In this case, dof should be set to a two-element vector,
                      which gives the scale range that was averaged together.
                      e.g. if one scale-averaged scales between 2 and 8,
                      then dof=[2,8]
        lag1  : LAG 1 Autocorrelation, used for signif levels. (optional)
                Default is 0.
        siglvl : Significance level to use. (optional)
                 Default is 0.95
        dof   : degrees-of-freedom for sgnif test
            Note: IF SIGTEST=1, then DOF can be a vector (same length as SCALEs),
            in which case NA is assumed to vary with SCALE.
            This allows one to average different numbers of times
            together at different scales, or to take into account
            things like the Cone of Influence.
            See discussion following Eqn (23) in Torrence & Compo.
            
    Outputs
        signif : significance levels as a function of scale
        
    """
    if np.atleast_1d(y) == 1:
        var = y
    else:
        var = np.var(y)
    
    j = len(scale)
    
    fourier_factor, dofmin, cdelta, gamma_fac, dj0 = motherparam(mother,param)
    period = scale*fourier_factor
    freq = dt/period
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
        if len(np.atleast_1d(dof)) != 1:
            pass
        elif dof == -1:
            dof = np.zeros(j)+dofmin
        else:
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
        if cdelta == -1:
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
    """
    Chisqure_solve
    
    Return the difference between calculated percentile and P.
    
    Written January 1998 by C. Torrence
    """
    pguess = gammainc(v/2,v*xguess/2)
    pdiff = np.abs(pguess - p)
    if pguess >= 1-1e-4:
        pdiff = xguess
    return pdiff

#def wave_coherency(wave1,time1,scale1,wave2,time2,scale2,dt=False,dj=False):
    
def waveletplot(wave,time,period,coi,sig95,levels=[0,2,5,10,20],
                cmap=False,title=False,xlabel=False,ylabel=False):
    
    plt.figure(figsize=(10,2))
    if not cmap:
        cmap=plt.cm.gray_r
    power=np.abs(wave)**2
    cs=plt.contourf(time,period,power,len(levels),cmap=cmap)
    im=plt.contourf(cs,levels=levels,cmap=cmap)
    plt.contour(time,period,sig95,[-99,1],color='k')
    plt.fill_between(time,coi,period.max(),color='grey',alpha=0.4,hatch='x')
    plt.ylim([period.max(),period.min()])
    plt.yscale('log',basey=2)
    if not xlabel:
        xlabel='Time'
    if not ylabel:
        ylabel='Period'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax=plt.gca().yaxis
    ax.set_major_formatter(sf())
    plt.ticklabel_format(axis='y',style='plain')
    if not title:
        title='Figure'
    plt.title(title)
    plt.colorbar(im)
    plt.grid()