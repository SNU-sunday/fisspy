from __future__ import absolute_import, division
import numpy as np

__all__ = ["get_lineName", "get_centerWV", "get_pure", "get_sel", "get_Inoise", "get_Linecenter", "get_photoLineWV"]

def get_lineName(cwv):
    """
    Get name of the spectrum

    Parameter
    ---------
    cwv: `float`
        Centerl wavelength of the spectrum

    Return
    ------
    line: `str`
        Spectral line designation.
    """
    if (cwv > 6550) * (cwv < 6570):
        line = 'ha'
    elif (cwv > 8530) * (cwv < 8550):
        line = 'ca'
    elif (cwv > 5880) * (cwv < 5900):
        line = 'na'
    elif (cwv > 5425) * (cwv < 5445):
        line = 'fe'
    return line

def get_centerWV(line):
    """
    Get the central wavelength of a line

    Parameter
    ---------
    line: `str`
        Spectral line designation.
        One among 'Ha', 'Ca', 'Na', 'Fe'.

    Return
    ------
    cwv: `float`
        Laboratory wavelength of the line.
    """
    ll = line.lower()
    if ll == 'ha':
        cwv = 6562.817
    elif ll == 'ca':
        cwv = 8542.091
    elif ll == 'na':
        cwv = 5889.951
    elif ll == 'fe':
        cwv = 5434.524

    return cwv 

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
    ndim = prof.ndim
    if ndim == 1:
        return _get_Linecenter1D(wv, prof, nd)
    elif ndim == 2:
        return _get_Linecenter2D(wv, prof, nd)

def _get_Linecenter1D(wv, prof, nd):
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
    s = prof[nd:-nd].argmin()+nd
    prof1 = prof[s-nd:s+nd+1]
    wv1 = wv[s-nd:s+nd+1]
    coeff = np.polyfit(wv1, prof1, 2)
    return -coeff[1]/(2*coeff[0])

def _get_Linecenter2D(wv, prof, nd):
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
    s = (prof[:,nd:-nd].argmin(1)+nd)[:,None]
    xx = np.arange(prof.shape[1])
    wh = (xx >= s-nd) * (xx<s+nd+1)
    prof1 = prof[wh].reshape((prof.shape[0], 2*nd+1))
    dwv = wv[1] - wv[0]
    ww = np.arange(-nd, nd+1)*dwv
    coeff = np.polyfit(ww, prof1.T, 2)
    return -coeff[1]/(2*coeff[0]) + wv[s[:,0]]

def get_pure(wv, line):
    """
    Determine whether blending by weak lines is absent or not at the specified wavelength(s)

    Parameters
    ----------
    wv: `~numpy.ndarray`
        Absolute wavelength(s) in unit of Angstrom
    line: `str`
        Spectral line designation.
        One among 'Ha', 'Ca', 'Na', 'Fe'.
    
    Return
    ------
    pure : `~numpy.ndarray`
        True if blending is not serious.
    """
    ll = line.lower()
    if ll == 'ha':
        hw = 0.15
        pure = (abs(wv-(6562.82-4.65)) > hw) * (abs(wv-(6562.82-3.20)) > hw) \
             * (abs(wv-(6562.82-2.55)) > hw) * (abs(wv-(6562.82-2.23)) > hw) \
             * (abs(wv-(6562.82+2.65)) > hw) * (abs(wv-(6562.82+2.69)) > hw) \
             * (abs(wv-(6562.82+3.80)) > hw) 
               
    elif ll == 'ca':
        hw = 0.15
        pure = (abs(wv-(8542.09-5.95)) > hw) * (abs(wv-8536.45) > hw) \
             * (abs(wv-8536.68) > hw) * (abs(wv-8537.95) > hw) \
             * (abs(wv-8538.25) > hw) * (abs(wv-(8542.09+2.2)) > hw) \
             * (abs(wv-8542.09+2.77) > hw) \
             * (abs(wv-wv.max()) > hw) * (abs(wv-wv.min()) > hw)
    else:
        raise ValueError("Line should be one of 'Ha' or 'Ca'")
    return pure 

def get_sel(wv, line):
    """
    To determine whether the data are to be selected or not for fitting

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths.
    line : `str`
        line designation.

    Returns
    -------
    sel : `numpy.ndarray`
        Boolean array. True if selected for fitting.

    """
    sel = get_pure(wv + get_centerWV(line), line=line)
    if line.lower() == 'ha':
        sel = sel*(abs(wv)<3.)
    else:    
        sel = sel*(abs(wv)<3.)
    return sel

def get_Inoise(intensity, line='ha'):
    """
    To get the noise level of intensity

    Parameters
    ----------
    intensity : `float` or `numpy.ndarray`
        intensities normalized by continuum.
    line : `str`, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    Inoise : `float` `or `numpy.ndarray`
        standard noises.
    """

    if line.lower() == 'ha':
        sigma0 = 0.01 
    elif line.lower() == 'ca':
        sigma0 = 0.01
    Inoise = sigma0*np.sqrt(intensity) 
    return Inoise
    
def get_photoLineWV(line, wvmin, wvmax):
    """
    To specicy the spectral line used to determine photospheric velocity 

    Parameters
    ----------
    line : `str`
        spectral band designation.
    wvmin : `float`
        minimum wavelength of the spectral band.
    wvmax : `float`
        maximum wavelength of the spectral band.

    Returns
    -------
    wvp : `float`
        laboratory wavelength of the photosperic line.
    dwv : `float`
        half of the wavelength range to be used 

    """
    if line.lower() == 'ha':
        wvp, dwv = 6559.580, 0.25
    if line.lower() == 'ca':
        wvp,dwv = 8536.165, 0.25 
        if (wvp > (wvmin+2*dwv))*(wvp < (wvmax-2*dwv)): return wvp, dwv
        wvp,dwv = 8548.079*(1+(-0.62)/3.e5), 0.25 
        
    return wvp, dwv


