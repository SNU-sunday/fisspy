from __future__ import absolute_import, division
import numpy as np
from ..align import alignOffset
from .get_inform import get_lineName, get_centerWV, get_pure, get_Linecenter, get_Inoise
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from multiprocessing import cpu_count

__author__ = "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"
__all__ = ["get_InstShift", "wvCalib", "wvCalib_simple", "smoothingProf", "corSLA", "corStrayLight", "corAsymmetry", "get_TauH2O", "get_Tlines", "corTlines", "get_TauS", "get_Sline", "corSline", "Voigt", "wvRecalib", "corAll", "normalizeProfile"]

def Voigt(u, a):
    """
    To determine the Voigt function

    Parameters
    ----------
    u : `numpy.ndarray`
        normalized and centered wavelengths
    a : `float`
        dimensionless damaping parameters

    Returns
    -------
    value
        function value

    """
    z = u + 1j*a
    return  wofz(z).real

def get_InstShift(data, refSpec, dw):
    """
    Get offsert value of the instrumental shift caused by the seeing and vibration of the spectrograph.

    Parameters
    ----------
    data: `~numpy.ndarray`
        N-d data.
        The last dimension should be the wavelength.
    refSpec: `~numpy.ndarray`
        1-D Reference spectrum.
    dw: `float`
        Wavelength scale in unit of angstrom.

    Returns
    -------
    woffset: `~numpy.ndarray`
        Offset values.
    """
    ndim = data.ndim
    sh = data.shape
    if ndim >= 4:
        raise ValueError("The dimension of the data should be less than 4.")
    else:
        refSpec2der = np.gradient(np.gradient(refSpec))
        refSpec2der[:2] = refSpec2der[-2:] = 0
        if ndim == 3:
            refSpec2der = refSpec2der * np.ones(sh[1:])
            data2der = np.gradient(np.gradient(data.mean(0), axis=1), axis=1)
            data2der[:,:2] = data2der[:,-2:] = 0
        elif ndim == 2:
            refSpec2der = refSpec2der * np.ones(sh)
            data2der = np.gradient(np.gradient(data, axis=1), axis=1)
            data2der[:,:2] = data2der[:,-2:] = 0
        elif ndim == 1:
            refSpec2der = refSpec2der * np.ones((4, sh[0]))
            data2der = np.gradient(np.gradient(data))
            data2der[:2] = data2der[-2:] = 0
            data2der = data2der * np.ones((4, sh[0]))
        yoff, xoff = alignOffset(data2der, refSpec2der)
        woffset = xoff*dw
    return woffset

def wvCalib(profile, h, method='simple'):
    """
    Wavelength calibration.

    Paramters
    ---------
    profile: `~numpy.ndarray`
        Spectrum
    h: `~astropy.io.fits.header.Header`
        Header of the spectrum.
    method: `str`
        Method to calibrate wavelength.
        'simple': calibration with the information of the header.
        'center': calibration with the center of the main line.
        'photo': calibration with the photospheric line and the main line.
        Default is 'simple'.

    Returns
    -------
    wv: `~numpy.ndarray`
        Wavelength.
    """
    if method == 'simple':
        wv = wvCalib_simple(h)
    elif method == 'center':
        wv = wvCalib_w_center(profile, h)
    elif method == 'photo':
        wv = wvCalib_w_photo(profile, h)
    return wv

def wvCalib_simple(h):
    """
    Wavelength calibration with the information of the header.

    Paramters
    ---------
    h: `~astropy.io.fits.header.Header`
        Header of the spectrum.
    
    Returns
    -------
    wv: `~numpy.ndarray`
        Wavelength.
    """
    nwv = h['naxis1']
    cwv = h['crval1']
    pcwv = h['crpix1']
    dwv = h['cdelt1']

    wv = (np.arange(nwv)-pcwv)*dwv+cwv
    return wv

def wvCalib_w_center(profile, h):
    """
    Wavelength calibration with the line center.

    Paramters
    ---------
    profile: `~numpy.ndarray`
        Spectrum
    h: `~astropy.io.fits.header.Header`
        Header of the spectrum.
    
    Returns
    -------
    wv: `~numpy.ndarray`
        Wavelength.
    """
    cwv = h['crval1']
    dwv = h['cdelt1']
    nwv = h['naxis1']
    nd = 5
    name = get_lineName(cwv)

    line = get_centerWV(name)
    
    wpix = np.arange(nwv)
    iwv = wvCalib_simple(h)
    w0 = iwv[profile.argmin()]
    wh = abs(iwv - w0) <= 0.3
    wp = get_Linecenter(wpix[wh], profile[wh], nd)
    wv = dwv * (wpix - wp) + line
    return wv

def wvCalib_w_photo(profile, h):
    """
    Wavelength calibration with the photospheric line and the line center.

    Paramters
    ---------
    profile: `~numpy.ndarray`
        Spectrum
    h: `~astropy.io.fits.header.Header`
        Header of the spectrum.
    
    Returns
    -------
    wv: `~numpy.ndarray`
        Wavelength.
    """
    cwv = h['crval1']
    nwv = h['naxis1']
    nds = [2,5]
    name = get_lineName(cwv)
    
    if name == 'Ha':
        lines = [6559.567, 6562.817]
        nds[1] = 6
    elif name == 'Ca':
        lines = [8536.165, 8542.091]
    elif name == 'Na':
        raise ValueError('This version cannot support the wvcalib for the 5889 line.')
    elif name == 'Fe':
        raise ValueError('This version cannot support the wvcalib for the 5434 line.')
    
    wps = np.zeros(2, dtype=float)
    iwv = wvCalib_simple(h)
    w0 = iwv[profile.argmin()]
    dw = lines[1] - w0
    wpix = np.arange(nwv)

    for i in range(len(lines)):
        wh = abs(iwv - lines[i] + dw) <= 0.3
        wps[i] = get_Linecenter(wpix[wh], profile[wh], nds[i])

    dwv = (lines[1]-lines[0])/(wps[1]-wps[0])
    wv = dwv*(np.arange(nwv) - wps[0]) + lines[0]
    return wv

def wvRecalib(wv, prof, line='ha'):
    """
    To recalibrate wavelengths of a spectrum 
    
    Parameters
    ------------
    wv:  array_lke
        wavelengths
    prof: array_like
        intensities
        
    line: str, optional (default='Ha')
        designation of the spectral band
        
    Returns    
    ------------
    wvnew:  array_like
        new wavelengths
    """
 
    if line.lower() == 'ha':
        wvline1 = 6559.580 #  Ti II
        dn1=2
        wvline2 = 6562.817   # H alpha
        dn2=3
        dispersion = 0.01905 # A/pix
        method = 1
    elif line.lower() =='ca':    
        wvline2 =  8536.165 # Si I 
        dn1=2
        wvline1 =  8542.091   # Ca II
        dn2=2
        dispersion = -0.02575 # A/pix
        method = 2
    else:
        raise ValueError("Unsupported line. Use 'ha' or 'ca'.")
        
    if method == 1:  
#        
#  Use two lines to re-determine both dispersion and wavelength reference        
        pline = abs(wv-wvline1) <= 0.3  
        wpix = np.arange(0., len(wv))
        wpix1 = get_Linecenter(wv[pline], prof[pline], nd=dn1)        
        pline = abs(wv-wvline2) <= 0.3   
        wpix2 = get_Linecenter(wv[pline], prof[pline], nd=dn2)       
        a=(wvline1-wvline2)/(wpix1-wpix2)  # dispersion
        b=(wpix1*wvline2-wpix2*wvline1)/(wpix1-wpix2)
        wvnew = a*wv+ b  
        
    if method == 2:
#   
#  Use one line to re-determine the wavelength reference  and 
#    and use the given dispersion     
        pline = abs(wv-wvline1) <= 0.3                     
        wpix = np.arange(0., len(wv)) 
        wpix1 = get_Linecenter(wpix[pline], prof[pline], nd=dn1)
        wvnew = dispersion*(wpix-wpix1)+wvline1
        
    return wvnew


def smoothingProf(data, method='savgol', **kwargs):
    """
    Parameters
    ----------
    data: `~numpy.ndarray`
        n-dimension spectral data.
        The last axis should be the wavelength domain.
    method: `str`, optional
        If 'savgol', apply the Savitzky-Golay noise filter in the wavelength axis.
        If 'gauss', apply the Gaussian noise filter in the wavelength axis.
        Default is 'savgol'.

    Other Parameters
    ----------------
    **kwargs : `~scipy.signal.savgol_filter` properties or `~scipy.ndimage.gaussian_filter1d` properties.

    See also
    --------
    `~scipy.signal.savgol_filter`.
    `~scipy.ndimage.gaussian_filter1d`

    Return
    ------
    sdata: `~numpy.ndarray`
        Smoothed data.
    """
    ndim = data.ndim
    axis = ndim-1
    
    if method == 'savgol':
        window_length = kwargs.pop('window_length', 7)
        polyorder = kwargs.pop('polyorder', 2)
        deriv = kwargs.pop('deriv', 0)
        delta = kwargs.pop('delta', 1.0)
        mode = kwargs.pop('mode', 'interp')
        cval = kwargs.pop('cval', 0.0)

        return savgol_filter(data, window_length, polyorder, deriv=deriv,
                            delta=delta, cval=cval, mode=mode, axis=axis)
    elif method == 'gauss':
        sigma = kwargs.pop('sigma', 1)
        return gaussian_filter1d(data, sigma, axis=axis, **kwargs)
    else:
        raise ValueError("Input one of 'savgol' or 'gauss'")

def corSLA(wv, data, refProf, line, pure=None, eps=0.027, zeta=0.055):
    """
    Correction of spectral line(s) profile for stray linght and far wing red-blue asymmetry.

    Parameters
    ----------
    wv: `~numpy.ndarray`, shape (N,)
        Absolute wavelengths in unit of Angstrom.
    data: '~numpy.ndarray`, shape (...,N)
        Line profile(s) to be corrected.
    refProf: `numpy.ndarray`, shape (N,)
        (Spatially averaged) Reference line profile.
    line: `str`
        Spectral line designation.
        One among 'Ha', 'Ca', 'Na', 'Fe'.
    pure: `~numpy.ndarray`
        True if not blended.
        Please see `~fisspy.correction.get_inform.Pure`
    eps: `float`
        Fraction of spatial stray light.
        The default is 0.027
    zeta: `float`
        Fration of spectral stray light.
        The default is 0.055

    
    Return
    ------
    I: `~numpy.ndarray`, shape (..., N)
        Correcteed line profile for stray light and far wing red-blue asymmetry.

    See Also
    --------
    Chae et al. (2013), https://ui.adsabs.harvard.edu/abs/2013SoPh..288....1C/abstract
    CorStrayLight: correction for stray light.
    CorAsymmetry: correction for far wing red-blue asymmetry.
    """
    if pure is None:
        pp = get_pure(wv, line)
    else:
        pp = pure
    I = corStrayLight(wv, data, refProf, line , pp, eps, zeta)
    I = corAsymmetry(wv, I, line, pp)

    return I

def corStrayLight(wv, data, refProf, line, pure=None, eps=0.027, zeta=0.055):
    """
    Correction of spectral line(s) profile for stray linght.

    Parameters
    ----------
    wv: `~numpy.ndarray`, shape (N,)
        Absolute wavelengths in unit of Angstrom.
    data: '~numpy.ndarray`, shape (...,N)
        Line profile(s) to be corrected.
    refProf: `numpy.ndarray`, shape (N,)
        (Spatially averaged) Reference line profile.
    line: `str`
        Spectral line designation.
        One among 'Ha', 'Ca', 'Na', 'Fe'.
    pure: `~numpy.ndarray`
        True if not blended.
        Please see `~fisspy.correction.get_inform.Pure`
    eps: `float`
        Fraction of spatial stray light.
        The default is 0.027
    zeta: `float`
        Fration of spectral stray light.
        The default is 0.055

    
    Return
    ------
    I: `~numpy.ndarray`, shape (..., N)
        Correcteed line profile for stray light.

    See Also
    --------
    Chae et al. (2013), https://ui.adsabs.harvard.edu/abs/2013SoPh..288....1C/abstract
    """
    if pure is None:
        pp = get_pure(wv, line)
    else:
        pp = pure
    cwv = get_centerWV(line)
    if line.lower() == 'ha':
        w = 4
    elif line.lower() == 'ca':
        w = 5
    else:
        w = 4
    wh_IC_obs = pp*(abs(wv-cwv-w) < 0.2) # blue far wing continuum
    IC_obs = data[...,wh_IC_obs].mean(-1) # find non-blending continuum
    IC_obs0 = refProf[wh_IC_obs].mean()

    # correcting for stray light (See equation 15 and 16 in Chae et al. 2013)
    # assume I_{c,obs}(0) \sim I_{c}(0)
    # data = I_{lambda,obs} 
    IC = (IC_obs[...,None]/IC_obs0-eps)/(1-eps)*IC_obs0
    I = (data/IC_obs[...,None] - zeta)/(1-zeta)*IC

    return I

def corAsymmetry(wv, data, line, pure=None):
    """
    Correction of spectral line(s) profile for far wing red-blue asymmetry.

    Parameters
    ----------
    wv: `~numpy.ndarray`, shape (N,)
        Absolute wavelengths in unit of Angstrom.
    data: '~numpy.ndarray`, shape (...,N)
        Line profile(s) to be corrected.
    line: `str`
        Spectral line designation.
        One among 'Ha', 'Ca', 'Na', 'Fe'.
    pure: `~numpy.ndarray`
        True if not blended.
        Please see `~fisspy.correction.get_inform.Pure`

    
    Return
    ------
    I: `~numpy.ndarray`, shape (..., N)
        Correcteed line profile for sfar wing red-blue asymmetry.
    """
    if pure is None:
        pp = get_pure(wv, line)
    else:
        pp = pure
    cwv = get_centerWV(line)

    # flattening
    sh = data.shape
    nw = sh[-1]
    na = int(data.size/nw)
    I = data.reshape((na,nw)).T

    # correcting for far blue-red wings asymmetry
    wh_IC_red = pp * (abs(wv-cwv) > 3.9)*(abs(wv-cwv) < 4.5)
    coeff = np.polyfit(wv[wh_IC_red], I[wh_IC_red], 1)
    p = np.polyval(coeff, wv[:,None])
    p /= np.maximum(p.mean(axis=0), 3e-2)[None,:]
    cI = (I/p).T.reshape(sh)

    return cI


def get_TauH2O(wv, rtau, dwv, line='ha'):
    """
    To determine the profile of water vapor optical thickness of the Earth's atmosphere
    

    Parameters
    ----------
    wv : `numpy.ndarray`
        wavelengths (1D)
    rtau : `float` or `numpy.ndarray`
        relative optical thickness (N dimensional array)
    dwv : `float` or `numpy.ndarray`
        wavelength offset of the optical thickness profile (N dimensional array)
    line : `str`, optional
        spectral line designation. The default is 'ha'.

    Returns
    -------
    tau
        optical thickness profile

    """

    if line.lower() == 'ha':
        wvHaline = get_centerWV(line)
        wvlines = np.array([6568.149, 6558.64, 6560.50, 6561.11, 6562.45, 
                 6563.521, 6564.055, 6564.206, 6565.53]) - wvHaline + 0.020
        tau0 = np.array([7., 1.5, 7.5, 4.5, 3.,  5., 4.2, 14.7, 1.5])/100.*2            
        a = 1.4
        w = 0.022
    elif line.lower() == 'ca':
        wvCaline = get_centerWV(line)
        wvlines = np.array([8539.895, 8540.817, 8546.22]) - wvCaline + 0.032
        tau0 = np.array([3.3, 8., 3.4])/100.*2.0       
        a = 1.9
        w = 0.034   
    
    taunormal = 0.
    nlines = len(wvlines)
    V0 = Voigt(0.,a) 

    if type(dwv) == np.ndarray:
        wvlines = wvlines-dwv[...,None]
        for pline in range(nlines):        
            taunormal += tau0[pline]*Voigt((wv-wvlines[...,pline,None])/w, a)/V0
        tau = rtau[...,None]*taunormal
    else:
        wvlines -= dwv
        for pline in range(nlines):        
            taunormal += tau0[pline]*Voigt((wv-wvlines[pline])/w, a)/V0
        tau = rtau*taunormal

    return tau

def resTauH2O(par, wave, profile, sigma, line):       
    tau = get_TauH2O(wave, par[0]**2, par[1], line=line) 
    res= np.convolve(profile*np.exp(tau), np.array([-1, 1.]), mode='same')/np.sqrt(2.)
    resD = res[1:-2]/sigma[1:-2] 
    resP = np.array([(par[0]-0.)/1., par[1]/0.01])
    res  = np.append(resD/np.sqrt(len(resD)), 0.01*resP/np.sqrt(len(resP)))                
    return res

def get_Tlines(wave, profile, line='ha', ncore=-1):
    """
    To determine the parameter relevant to the content of water vapor in the Earth's atmosphere

    Parameters
    ----------
    wave: `numpy.ndarray`
        wavelengths (1D)
    profile: `numpy.ndarray`
        intensities (1D or ND)
    line: `str`, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    par: `numpy.ndarray`
        Array of parameters for each profile (rtau, dwv)
    """
    
    ndim = profile.ndim

    if line.lower() == 'ha': 
        s = ((wave - 2.) * (wave - 1.) < 0)
    elif line.lower() == 'ca':
        s = ((wave + 1.7) * (wave + 1.) < 0)
    else:
        raise ValueError("Unsupported line. Use 'ha' or 'ca'.")

    wave1 = wave[s]
    
    def process_profile(p):
        profile1 = p[s] / p.max()
        sigma = get_Inoise(profile1, line=line)
        par0 = [0.5, 0.]
        res_lsq = least_squares(resTauH2O, par0, max_nfev=50, jac='2-point', args=(wave1, profile1, sigma, line))
        return res_lsq.x
    
    if ndim == 1:
        return process_profile(profile)
    
    else:
        shape = profile.shape[:-1]  # 마지막 축 제외한 모양
        profile_reshaped = profile.reshape(-1, profile.shape[-1])  # 2D 형태로 변환
        results = Parallel(n_jobs=ncore, backend='loky')(
            delayed(process_profile)(p) for p in profile_reshaped
        )
        return np.array(results).reshape(*shape, -1)  # 원래 모양으로 복구

def corTlines(wave, profile, par, line='ha'):
    tau = get_TauH2O(wave, par[...,0]**2, par[...,1], line=line)
    return profile*np.exp(tau)

def get_TauS(wv, rtau, dwv):
    """
    To determine the optical thickness profile of the Co I line in the H alpha band

    Parameters
    ----------
    wv : `numpy.ndarray`
        wavelengths measured from the center of the H alpha line.
    rtau : `numpy.ndarray`
        relative otpical thickness 
    dwv : `numpy.ndarray`
        wavelength offset(s)

    Returns
    -------
    tau : `numpy.ndarray`
        optical thickness(es).

    """
        
    wvline =  0.593 + dwv        
    tau0 = 0.15   
    a = 0.00
    w = 0.12
    if type(dwv) == np.ndarray:
        u = (wv - wvline[...,None])/w
        taunormal = tau0*Voigt(u,a)/Voigt(0., a) #exp(-u**2)
        tau = rtau[...,None]*taunormal
    else:
        u = (wv - wvline)/w
        taunormal = tau0*Voigt(u,a)/Voigt(0., a) #exp(-u**2)
        tau = rtau*taunormal
    return tau

def resTauS(par, wave, profile, sigma):
    tau1 = get_TauS(wave, par[0]**2, par[1])
    res = np.convolve(profile*np.exp(tau1), np.array([-1.,2.,-1.]), mode='same')/np.sqrt(6.)
    resD = res[2:-3]/sigma[2:-3]  
    resP = np.array([(par[0]-0.)/1., (par[1]-0.)/0.01])
    res  = np.append(resD/np.sqrt(len(resD)), 0.01*resP/np.sqrt(len(resP)))     
    return res

def get_Sline(wv, profile, ncore=-1):
    """
    Optimized function to determine the optical thickness parameter of Co I line.

    Parameters
    ----------
    wv: `numpy.ndarray`
        Wavelengths (1D).
    profile: `numpy.ndarray`
        Intensities (1D or ND).

    Returns
    -------
    par: `numpy.ndarray`
        Array of parameters for each profile (rtau, dwv).
    """
    ndim = profile.ndim
    s = ((wv - 0.25) * (wv - 1.) < 0)  
    wv1 = wv[s]

    def process_Sprofile(p):
        if p.max() > 0.95:
            return np.array([0., 0.])
        
        profile1 = p[s] / p.max()
        sigma = get_Inoise(profile1, line='ha')
        par0 = [0.5, 0.]
        
        res_lsq = least_squares(resTauS, par0, max_nfev=50, jac='2-point', args=(wv1, profile1, sigma))
        par = res_lsq.x
        par[0] = abs(par[0])
        return par

    if ndim == 1:
        return process_Sprofile(profile)

    else:
        # 3D 이상 배열일 경우 2D로 변환하여 병렬 처리
        shape = profile.shape[:-1]  # 마지막 축 제외한 차원 저장
        profile_reshaped = profile.reshape(-1, profile.shape[-1])  # (N, M) 형태로 변환

        # 병렬 실행
        results = Parallel(n_jobs=ncore, backend='loky')(
            delayed(process_Sprofile)(p) for p in profile_reshaped
        )

        return np.array(results).reshape(*shape, -1)  # 원래 차원으로 복원

def corSline(wv, profile, par):
    """
    To correct the H alpha spectral profile for the Co I line blending 

    Parameters
    ----------
    wave: `numpy.ndarray`
        wavelengths (1D)
    profile: `numpy.ndarray`
        intensities (1D or ND)
    par: `numpy.ndarray`
        Co I line blending parameters (rtau, dwv)

    Returns
    -------
    profilenew : `numpy.ndarray`
        corrrected line profile.

    """
    tau = get_TauS(wv, par[...,0]**2, par[...,1])
    return profile * np.exp(tau)

def corAll(fissobj, subsec=None, ncore=-1):
    """
    subsect: `list`, optional
        Subsection of the data to be corrected.
        The default is None.
        [l, r, b, t]
    """
    nc = cpu_count()

    ncc = np.minimum(nc, ncore)
    if ncc == -1:
        ncc = nc
    if subsec is None:
        x1, x2, y1, y2 = [0, fissobj.nx, 0, fissobj.ny]
    else:
        x1, x2, y1, y2 = subsec

    nprof = (x2-x1) * (y2-y1)
    if nprof < 50:
        ncc = 1

    if fissobj.avp is None:
        normalizeProfile(fissobj)

    if nprof == 1:
        do = fissobj.data[y1, x1]
    else:
        do = fissobj.data[y1:y2, x1:x2]
    Tpar = get_Tlines(fissobj.Rwave, do, line=fissobj.line, ncore=ncc)
    d = corTlines(fissobj.Rwave, do, Tpar, line=fissobj.line)
    if fissobj.line.lower() == 'ha':
        spar = get_Sline(fissobj.Rwave, d, ncore=ncc)
        d = corSline(fissobj.Rwave, d, spar)
    d = corSLA(fissobj.Awave, d, refProf=fissobj.avp, line=fissobj.line, pure=fissobj.pure)

    return d

def normalizeProfile(fissobj):
    """
    Normalize the profile of the data
    """
    avp = fissobj.get_avProfile()
    fissobj.wave = wvRecalib(fissobj.wave, avp, fissobj.line)
    fissobj.Awave = fissobj.wave.copy()
    fissobj.Rwave = fissobj.Awave - fissobj.wvlab
    if fissobj.line.lower() == 'ha':
        rr = 1.01
    elif fissobj.line.lower() == 'ca':
        rr = 1.09
    refc = avp[abs(abs(fissobj.Rwave)-4.5)<0.05].mean()*rr
    fissobj.avp = avp/refc
    fissobj.data = fissobj.data/refc

    pure = get_pure(fissobj.Awave, fissobj.line)
    Tpar = get_Tlines(fissobj.Rwave, fissobj.avp, fissobj.line, ncore=1)
    fissobj.refProfile = corTlines(fissobj.Rwave, fissobj.avp, Tpar, fissobj.line)
    fissobj.refProfile = corSLA(fissobj.Awave, fissobj.refProfile, refProf=fissobj.avp, line=fissobj.line, pure=pure)
    fissobj.pure = pure
