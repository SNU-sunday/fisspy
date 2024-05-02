from __future__ import absolute_import, division

__all__ = ["lineName", "centerWV", "Pure"]

def lineName(cwv):
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
        line = 'Ha'
    elif (cwv > 8530) * (cwv < 8550):
        line = 'Ca'
    elif (cwv > 5880) * (cwv < 5900):
        line = 'Na'
    elif (cwv > 5425) * (cwv < 5445):
        line = 'Fe'
    return line

def centerWV(line):
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

def Pure(wv, line):
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

def BadSteps(FS):
    """
    """
    None