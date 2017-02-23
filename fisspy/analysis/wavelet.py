"""
Calculate the wavelet and its significance.
"""
from __future__ import division, absolute_import
import numpy as np
from scipy.special._ufuncs import gamma, gammainc
from scipy.optimize import fminbound as fmin
from scipy.fftpack import fft, ifft

__author__ = "Juhyeong Kang"
__email__ =  "jhkang@astro.snu.ac.kr"
__all__ = ['wavelet', 'iwavelet', 'motherfunc', 'motherparam',
           'wave_signif', 'chisquare_inv', 'chisquare_solve',
           'wave_coherency', 'fast_conv', 'fast_conv2']

def wavelet(y, dt,
            dj=0.25, s0=False, j=False,
            mother='MORLET', param=False, pad=True):
    """
    Compute the wavelet transform of the given y
    with sampling rate dt.
    
    By default, the MORLET wavelet (k0=6) is used.
    The wavelet basis is normalized to have total energy=1
    at all scales.
    
    Parameters
    ----------
    y : ~numpy.ndarray
        The time series of length n.
    dt : float
        The time step between each y values.
        i.e. the sampling time.
    dj : (optional) float
        The spacing between discrete scales.
        The smaller, the better scale resolution.
            * Default is 0.25
    s0 : (optional) float
        The smallest scale of the wavelet.  
            * Default is :math:`2 \cdot dt`.
    j : (optional) int
        The number of scales minus one.
        Scales range from :math:`s0` up to :math:`s_0\cdot 2^{j\cdot dj}`, to give
        a total of :math:`j+1` scales.
            * Default is :math:`j=\log_2{(\\frac{n dt}{s_0 dj})}`.
    mother : (optional) str
        The mother wavelet function.
        The choices are 'MORLET', 'PAUL', or 'DOG'
            * Default is **'MORLET'**
    param  : (optional) int
        The mother wavelet parameter.\n
        For **'MORLET'** param is k0, default is **6**.\n
        For **'PAUL'** param is m, default is **4**.\n
        For **'DOG'** param is m, default is **2**.\n
    pad : (optional) bool
        If set True, pad time series with enough zeros to get
        N up to the next higher power of 2.
        This prevents wraparound from the end of the time series
        to the beginning, and also speeds up the FFT's 
        used to do the wavelet transform.
        This will not eliminate all edge effects.
    
    Returns
    -------
    wave : ~numpy.ndarray
        The WAVELET transform of y.
        (j+1, n) complex arry.
        np.arctan2(wave.imag,wave.real) gives the WAVELET phase.
        wave.real gives the WAVELET amplitude.
        The WAVELET power spectrum is :math:`|wave|^2`.
    period : ~numpy.ndarray
        The vecotr of 'Fourier' periods (in time units)
        that correspods to the scales.
    scale : ~numpy.ndarray
        The vecotr of scale indices, given by :math:`s_0 \cdot 2^{j \cdot dj}, 
        j=0...j`
        where :math:`j+1` is the total number of scales.
    coi : ~numpy.ndarray
        The Cone-of-Influence, which is a vector of N points
        that contains the maximum period of useful information
        at that particular time.
        Periods greater than this are subject to edge effets.
    
    Notes
    -----
        This function based on the IDL code WAVELET.PRO written by C. Torrence, 
        and Python code waveletFuncitions.py written by E. Predybayalo.
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
    http://paos.colorado.edu/research/wavelets/
    
    Example
    -------
    >>> from fisspy.analysis import wavelet
    >>> wave, period, scale, coi = wavelet.wavelet(y,dt,dj=dj,j=j,mother=mother,pad=True)
    
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
    """
    Inverse the wavelet to get the time-series
    
    Parameters
    ----------
    wave : ~numpy.ndarray
        wavelet power.
    scale : ~numpy.ndarray
        The vecotr of scale indices, given by :math:`s_0 \cdot 2^{j \cdot dj}`
    dt : float
        The time step between each y values.
    dj : (optional) float
        The spacing between discrete scales.
        The smaller, the better scale resolution.
            * Default is 0.25
    mother : (optional) str
        The mother wavelet function.
        The choices are 'MORLET', 'PAUL', or 'DOG'
            * Default is **'MORLET'**
    param : (optional) int
        The mother wavelet parameter.\n
        For **'MORLET'** param is k0, default is **6**.\n
        For **'PAUL'** param is m, default is **4**.\n
        For **'DOG'** param is m, default is **2**.\n
    
    Returns
    -------
    iwave : ~numpy.ndarray
        Inverse wavelet.
    
    Notes
    -----
        This function based on the IDL code WAVELET.PRO written by C. Torrence, 
        and Python code waveletFuncitions.py written by E. Predybayalo.
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
    http://paos.colorado.edu/research/wavelets/
        
    Example
    -------
    >>> from fisspy.analysis import wavelet
    >>> iwave=wavelet.iwavelet(wave,scale,dt)
    """
    a, b = wave.shape
    c = len(scale)
    scale2=1/scale**0.5
    mother=mother.upper()
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
    Compute the Fourier factor and period.
    
    Parameters
    ----------
    mother : str
        A string, Equal to 'MORLET' or 'PAUL' or 'DOG'.
    k : 1d ndarray
        The Fourier frequencies at which to calculate the wavelet.
    scale : ~numpy.ndarray
        The wavelet scale.
    param : int
        The nondimensional parameter for the wavelet function.
    
    Returns
    -------
    nowf : ~numpy.ndarray
        The nonorthogonal wavelet function.
    period : ~numpy.ndarrary
        The vecotr of "Fourier" periods (in time units)
    fourier_factor : float
        the ratio of Fourier period to scale.
    coi : int
        The cone-of-influence size at the scale.
    
    Notes
    -----
        This function based on the IDL code WAVELET.PRO written by C. Torrence, 
        and Python code waveletFuncitions.py written by E. Predybayalo.
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
    http://paos.colorado.edu/research/wavelets/
        
    Example
    -------
    >>> nowf, period, fourier_factor, coi = motherfunc(mother,k, scale,param)
    
    """
    mother=mother.upper()
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
    """
    Get the some values for given mother function of wavelet.
    
    Parameters
    ----------
    mother : str
    param : int
        The nondimensional parameter for the wavelet function.
        
    Returns
    -------
    fourier_factor : float
        the ratio of Fourier period to scale.
    dofmin : float
        Degrees of freedom for each point in the wavelet power.
        (either 2 for MORLET and PAUL, or 1 for the DOG)
    cdelta : float
        Reconstruction factor.
    gamma_fac : float
        decorrelation factor for time averaging.
    dj0 : float
        factor for scale averaging.
    
    Notes
    -----
        This function based on the IDL code WAVELET.PRO written by C. Torrence, 
        and Python code waveletFuncitions.py written by E. Predybayalo.
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
    http://paos.colorado.edu/research/wavelets/
    
    Example
    -------
    >>> fourier_factor, dofmin, cdelta,gamma_fac, dj0 = motherparam(mother,param)
        
    """
    mother=mother.upper()
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
    Compute the significance levels for a wavelet transform.
    
    Parameters
    ----------
    y : float or ~numpy.ndarray
        The time series, or the variance of the time series.
        If this is a single number, it is assumed to be the variance.
    dt : float
        The sampling time.
    scale : ~numpy.ndarray
        The vecotr of scale indices, from previous call to WAVELET.
    sigtest : (optional) int
        Allowable values are 0, 1, or 2
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
    lag1 : (optional) float
        LAG 1 Autocorrelation, used for signif levels.
            * Default is 0.
    siglvl : (optional) float
        Significance level to use.
            * Default is 0.95
    dof : (optional) float
        degrees-of-freedom for sgnif test.
            * Default is -1, and it means the False.

            
    Returns
    -------
        signif : ~numpy.ndarray
            Significance levels as a function of scale.
        
    Notes
    -----
    IF SIGTEST=1, then DOF can be a vector (same length as SCALEs), 
    in which case NA is assumed to vary with SCALE. 
    This allows one to average different numbers of times 
    together at different scales, or to take into account 
    things like the Cone of Influence.\n
    See discussion following Eqn (23) in Torrence & Compo.\n
    This function based on the IDL code WAVE_SIGNIF.PRO written by C. Torrence, 
    and Python code waveletFuncitions.py written by E. Predybayalo.
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
    http://paos.colorado.edu/research/wavelets/
    
    Example
    -------
    >>> signif=wavelet.wave_signif(y,dt,scale,2,mother='morlet',dof=[s1,s2],gws=gws)
    
    """
    if len(np.atleast_1d(y)) == 1:
        var = y
    else:
        var = np.var(y)
    
    j = len(scale)
    
    fourier_factor, dofmin, cdelta, gamma_fac, dj0 = motherparam(mother,param)
    period = scale*fourier_factor
    freq = dt/period
    try:
        len(gws)
        fft_theor = gws.copy()
    except:
        fft_theor = (1-lag1**2)/(1-2*lag1*np.cos(freq*2*np.pi)+lag1**2)
        fft_theor*=var


    signif = fft_theor.copy()
    
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
        avg = (period>=s1)*(period<=s2)
        navg = avg.sum()
        if not navg:
            raise ValueError('No valid scales between %s and %s' %(repr(s1),repr(s2)))
        s1=scale[avg].min()
        s2=scale[avg].max()
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
    Inverse of chi-square cumulative distribution function(CDF).
    
    Parameters
    ----------
    p : float
        probability
    v : float
        degrees of freedom of the chi-square distribution
    
    Returns
    -------
    x : float
        the inverse of chi-square cdf
        
    Example
    -------
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

def wave_coherency(wave1,time1,scale1,wave2,time2,scale2,
                   dt=False,dj=False,coi=False,nosmooth=False):
    """
    Compute the wavelet coherency between two time series.
    
    Parameters
    ----------
    wave1 : ~numpy.ndarray
        Wavelet power spectrum for time series 1.
    time1 : ~numpy.ndarray
        A vector of times for time series 1.
    scale1 : ~numpy.ndarray
        A vector of scales for time series 1.
    wave2 : ~numpy.ndarray
        Wavelet power spectrum for time series 2.
    time2 : ~numpy.ndarray
        A vector of times for time series 2.
    scale2 : ~numpy.ndarray
        A vector of scales for time series 2.
    dt : (optional) float
        Amount of time between each Y value, i.e. the sampling time.
            If not input, then calculated from time1[1]-time1[0]
    dj : (optional) float
        The spacing between discrete scales.
            If not input, then calculated from scale1
    coi : (optional) ~numpy.ndarray
        The array of the cone-of influence.
    nosmooth : (optional) bool
        If True, then just compute the global_coher, global_phase, and
        the unsmoothed cross_wavelet and return.
    
    Returns
    -------
    result : dict
        The result is dictionary has these information.
        
        cross_wavelet : ~numpy.ndarray
            The cross wavelet between the time series.
        time : ~numpy.ndarray
            The time array given by the overlap of time1 and time2.
        scale : ~numpy.ndarray
            The scale array of scale indices, given by the overlap of 
            scale1 and scale2.
        wave_phase : ~numpy.ndarray
            The phase difference between time series 1 and time series 2.
        wave_coher : ~numpy.ndarray
            The wavelet coherency, as a function of time and scale.
        global_phase : ~numpy.ndarray
            The global (or mean) phase averaged over all times.
        global_coher : ~numpy.ndarray
            The global (or mean) coherence averaged over all times.
        power1 : ~numpy.ndarray
            The wavelet power spectrum should be the same as wave1
            if time1 and time2 are identical, otherwise it is only the
            overlapping portion. If nosmooth is set,
            then this is unsmoothed, otherwise it is smoothed.
        power2 : ~numpy.ndarray
            same as power 1 but for time series 2.
        coi : ~numpy.ndarray
            The array of the cone-of influence.
        
    Notes
    -----
        This function based on the IDL code WAVE_COHERENCY.PRO written by C. Torrence, 
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
    http://paos.colorado.edu/research/wavelets/
    
    Example
    -------
    >>> res=wavelet.wave_coherency(wave1,time1,scale1,wave2,time2,scale2,\
dt,dj,coi=coi)
    
    """
    if not dt: dt=time1[1]-time1[0]
    if not dj: dj=np.log2(scale1[1]/scale1[0])
    if time1 is time2:
        t1s=0
        t1e=len(time1)
        t2s=t1s
        t2e=t1e
    else:
        otime_start=min([time1.min(),time2.min()])
        otime_end=max([time1.max(),time2.max()])
        t1=np.where((time1 >= otime_start)*(time1 <= otime_end))[0]
        t1s=t1[0]
        t1e=t1[-1]+1
        t2=np.where((time2 >= otime_start)*(time2 <= otime_end))[0]
        t2s=t2[0]
        t2e=t2[-1]+1
    
    oscale_start=min([scale1.min(),scale2.min()])
    oscale_end=max([scale1.max(),scale2.max()])
    s1=np.where((scale1 >= oscale_start)*(scale1 <= oscale_end))[0]
    s2=np.where((scale2 >= oscale_start)*(scale2 <= oscale_end))[0]
    s1s=s1[0]
    s1e=s1[-1]+1
    s2s=s2[0]
    s2e=s2[-1]+1
    
    cross_wavelet=wave1[s1s:s1e,t1s:t1e]*wave2[s2s:s2e,t2s:t2e].conj()
    power1=np.abs(wave1[s1s:s1e,t1s:t1e])**2
    power2=np.abs(wave2[s2s:s2e,t2s:t2e])**2
    
    time=time1[t1s:t1e]
    scale=scale1[s1s:s1e]
    nj=s1e-s1s
    
    global1=power1.sum(1)
    global2=power2.sum(1)
    global_cross = cross_wavelet.sum(1)
    global_coher = np.abs(global_cross)**2/(global1*global2)
    global_phase = np.arctan(global_cross.imag/global_cross.real)*180./np.pi
    
    if nosmooth:
        result = dict(global_coherence=global_coher,global_phase=global_phase,
                      cross_wavelet=cross_wavelet)
        return result
    
    nt=(4*scale/dt)//2*4+1
    nt2=nt[:,np.newaxis]
    ntmax=nt.max()
    g=np.arange(ntmax)*np.ones((nj,1))
    wh=g >= nt2
    time_wavelet=(g-nt2//2)*dt/scale[:,np.newaxis]
    wave_func=np.exp(-time_wavelet**2/2)
    wave_func[wh]=0
    wave_func=(wave_func/wave_func.sum(1)[:,np.newaxis]).real
    cross_wavelet=fast_conv(cross_wavelet,wave_func,nt2)
    power1=fast_conv(power1,wave_func,nt2)
    power2=fast_conv(power2,wave_func,nt2)
    scales=scale[:,np.newaxis]
    cross_wavelet/=scales
    power1/=scales
    power2/=scales
    
    nw=int(0.6/dj/2+0.5)*2-1
    weight=np.ones(nw)/nw
    cross_wavelet=fast_conv2(cross_wavelet,weight)
    power1=fast_conv2(power1,weight)
    power2=fast_conv2(power2,weight)
    
    wave_phase=180./np.pi*np.arctan(cross_wavelet.imag/cross_wavelet.real)
    power3=power1*power2
    whp=power3 < 1e-9
    power3[whp]=1e-9
    wave_coher=(np.abs(cross_wavelet)**2/power3).real
    
    result=dict(cross_wavelet=cross_wavelet,time=time,scale=scale,
                wave_phase=wave_phase,wave_coher=wave_coher,
                global_phase=global_phase,global_coher=global_coher,
                power1=power1,power2=power2,coi=coi)
    return result





def fast_conv(f,g,nt):
    """
    Fast convolution two given function f and g (method 1)
    """
    nf=f.shape
    ng=g.shape
    npad=2**(int(np.log2(max([nf[1],ng[1]])))+1)
    wh1=np.arange(nf[0],dtype=int)
    wh2=np.arange(nf[1],dtype=int)*np.ones((nf[0],1),dtype=int)-(nt.astype(int)-1)//2-1
    pf=np.zeros([nf[0],npad],dtype=complex)
    pg=np.zeros([nf[0],npad],dtype=complex)
    pf[:,:nf[1]]=f
    pg[:,:ng[1]]=g
    conv=ifft(fft(pf)*fft(pg[:,::-1]))
    result=conv[wh1,wh2.T].T
    return result

def fast_conv2(f,g):
    """
    Fast convolution two given function f and g (method2)
    """
    nf=f.shape
    ng=len(g)
    npad=2**(int(np.log2(max([nf[0],ng])))+1)
    
    wh1=np.arange(nf[1],dtype=int)
    wh2=np.arange(nf[0],dtype=int)*np.ones((nf[1],1),dtype=int)+ng//2
    pf=np.zeros([npad,nf[1]],dtype=complex)
    pg=np.zeros([npad,nf[1]],dtype=complex)
    pf[:nf[0],:]=f
    pg[:ng,:]=g[:,np.newaxis]
    conv=ifft(fft(pf,axis=0)*fft(pg,axis=0),axis=0)
    result=conv[wh2.T,wh1]
    return result