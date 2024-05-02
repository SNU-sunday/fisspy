from __future__ import absolute_import, division
import numpy as np
from ..align import alignOffset
from .get_inform import lineName, centerWV, Pure
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

__author__ = "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"
__all__ = ["get_InstShift", "get_Linecenter", "wvCalib", "smoothingProf", "corSLA", "corStrayLight", "corAsymmetry"]

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
    name = lineName(cwv)

    line = centerWV(name)
    
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
    name = lineName(cwv)
    
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

def get_Linecenter(wv, prof, nd):
    """
    Get pixel of the line center.
    To determine the central wavelength of an absorption line using the 2-nd polynomial fitting of the line core.

    Parameters
    ----------
    wv: `~numpy.ndarray`
        Wavelength
    prof: `~numpy.ndarray`
        Spectrum
    nd: `int`, optional
        half number of data points.
        Default is 2.

    Returns
    -------
    value:  central wavelength
    """
    s=prof[nd:-nd].argmin()+nd
    prof1=prof[s-nd:s+nd+1]
    wv1=wv[s-nd:s+nd+1]            
    coeff=np.polyfit(wv1, prof1, 2)
    return -coeff[1]/(2*coeff[0])

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
        pp = Pure(wv, line)
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
        pp = Pure(wv, line)
    else:
        pp = pure
    cwv = centerWV(line)
    if line == 'Ha':
        w = 4
    elif line == 'Ca':
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
        pp = Pure(wv, line)
    else:
        pp = pure
    cwv = centerWV(line)

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

