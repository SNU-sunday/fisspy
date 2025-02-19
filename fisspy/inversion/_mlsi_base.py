from __future__ import absolute_import, division

import astropy.constants as const
from ..correction import get_centerWV, Voigt, get_Inoise, get_pure, get_sel, get_photoLineWV, get_Linecenter
import numpy as np
from ..read import FISS
from scipy.special import expn
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from interpolation.splines import LinearSpline
from joblib import Parallel, delayed
from numba import njit, jit, prange

def Penalty(x):
    """
    To deterine the penalaty function for positivity. 
    The penalty is zero for a positive input, and is equal to the negative input.  

    Parameters
    ----------
    x : `numpy.ndarray`
        input(s)

    Returns
    -------
    penlaty:  `numpy.ndarray`
        output penalty(s) 

    """
    penalty = x*(x<0)
    return penalty

def Trad(I, line, I0=1.):
    """
    Radiation temperature corresponding to intensity

    Parameters
    ----------
    I : `numpy.ndarray`
        intensity(s).
    line : `str`
        line designation.
    I0 : `float`, optional
        the disk center intensity in normalized unit. The default is 1..

    Returns
    -------
    Trad : array_like
        radiation temperature.
    """

    wv = get_centerWV(line)
    h = const.h.cgs.value
    c = const.c.cgs.value
    k = const.k_B.cgs.value
    I00 = 2*h*c**2/(wv*1e-8)**4/wv
    if line.lower() == 'ha':
        Icont = 2.84e6
    if line.lower() == 'ca':
        Icont = 1.76e6
    Ilambda = (I/I0)*Icont 
    hnuoverk = h*(c/(wv*1.e-8))/k # 1.43*10000./wv
    Trad =hnuoverk/(np.log(1+I00/Ilambda))
    return Trad

def Dwidth(T, xi):
    """
    To determine the Doppler widths of the H alpha line and Ca II 8542 line

    Parameters
    ----------
    T : `float` or `numpy.ndarray`
        hydrogen temperature(s)
    xi : `float` or `numpy.ndarray`
        nonthermal speed(s) in unit of km/s

    Returns
    -------
    DwHa : `float` or `numpy.ndarray`
        Doppler width(s) of the H alpha line in unit of Angstrom
    DwCa : `float` or `numpy.ndarray`
        Doppler width(s) of the Ca II 8542 line in unit of Angstrom
    """
    m = const.m_p.cgs.value
    k = const.k_B.cgs.value
    c = const.c.cgs.value * 1e-5
    DwHa = 6562.817*np.sqrt(xi**2+(2*k*T/m)/1e10)/c
    DwCa = 8542.091*np.sqrt(xi**2+(2*k*T/(40*m))/1e10)/c
    return DwHa, DwCa

def Dw2TnXi(DwHa, DwCa):
    """
    To determine hydrogen temperature and nonthermal speed from the Doppler widths
    of the H alpha line and the Ca II 8542 line

    Parameters
    ----------
    DwHa : `float` or `numpy.ndarray`
        Doppler width(s) of the H alphalline in unit of Angstrom
    DwCa : `float` or `numpy.ndarray`
        Doppler width(s) of the Ca II 8542 line in unit of Angstrom

    Returns
    -------
    T : `float` or `numpy.ndarray`
        hydgregen temperature(s) in unit of K
    xi : `float` or `numpy.ndarray`
        nonthermal speed(s) in unit of km/s
    """
    hwv = get_centerWV('ha')
    cwv = get_centerWV('ca')
    c = const.c.cgs.value
    k = const.k_B.cgs.value
    m = const.m_p.cgs.value
    yHa = (DwHa/hwv*c)**2
    yCa = (DwCa/cwv*c)**2    
    delt = np.array(yHa-yCa)
    delt[delt<0]=1.
    Temp = (1.-1./40.)**(-1.)*(m/k)/2.*delt
    
    delt = np.array(40*yCa - yHa)
    delt[delt<0]=1.
    xi = np.sqrt(delt/(40-1))/1e5   # to km/s
    return Temp, xi

def get_nBadSteps(f):
    """
    To determine the number of bad steps in the  FISS raster scan

    Parameters
    ----------
    f : `str`
        FISS file name.

    Returns
    -------
    nbad : `int`
        number of bad steps.
    Ic : `numpy.ndarray`
        2D array of continuum-like raster image (ny, nx).
    """
    fiss = FISS(f)
    fiss.data = fiss.data[:,::-1] # flip the image in the x direction
    Ic = (fiss.data[..., 50:60]).mean(2)
    s = Ic.shape
    a = Ic[:,:-1]
    amag = np.sqrt((a**2).mean(1))
    b = np.roll(Ic, -1, axis=1)[:,:-1]
    bmag = np.sqrt((b**2).mean(1))
    det = (a*b).mean(1) - 0.7*(amag*bmag)
    wh = det <= 0
    nbad = wh.sum()
    return  nbad, Ic

def parInform(nlayers=3):
    """
    To yield the description of the specified parameter in the three layer model. 
    
    The first 15 parameters (with indice 0 to 14) are the primary parameters 
    neccessary and sufficient to specify the model. The remaining parameters are 
    the secondary parameters that can be caluclated from the other parameters.

    Parameters
    ----------
    index : `int`
        index of the parameter.

    Returns
    -------
    descript : `str`
        decription of the parameter.

    """
    if nlayers == 3:
        lpar = [("vp", "Line-of-sight velocity at the photosphere in"),
                ("log eta", "The ratio of peak line absorption to continuum absorption"),
                ("log wp", "Doppler width at the photosphere in Angstrom"),
                ("log ap", "Dimensionless damping parameter at the photosphere"),
                ("log Sp", "Source function at the photosphere"),
                ("log S2", "Source function at the bottom of the chromosphere"),
                ("log tau2", "Optical thickness in the low chromosphere"),
                ("log tau1", "Optical thickness in the upper chromosphere"),
                ("v1", "Line-of-sight velocity in the middle of the chromosphere"),
                ("v0", "Line-of-sight velocity at the top of the choromosphere"),
                ("log w1", "Doppler width in the middle of the chromosphere"),
                ("log w0", "Doppler width at the top of the choromosphere"),
                ("log S1", "Source function in the middle of the chromosphere"),
                ("log S0", "Source function at the top of the choromosphere"),
                ("log wg", "123"),
                ("log epsD", "The goodness of the model for data requirement"),
                ("log epsP", "The goodness of the model for parameter requirements"),
                ("log Radloss2", "Radiative loss at the bottom of the chromosphere"),
                ("log Radloss1", "Radiative loss in the middle of the chromosphere"),
                ]
    return lpar

def apar2dpar(apar):
    dpar = {'vp':apar[0], 'log eta':apar[1], 'log wp':apar[2], 'log ap':apar[3], 'log Sp':apar[4], 'log S2':apar[5], 'log tau2':apar[6], 'log tau1':apar[7], 'v1':apar[8], 'v0':apar[9], 'log w1':apar[10], 'log w0':apar[11], 'log S1':apar[12], 'log S0':apar[13], 'log wg':apar[14]}
    return dpar

def parDefault(line='ha'):
    """
    Provide the default model parameters and their prior deviations  

    Parameters
    ----------
    line : `str`, optional
        line designation. The default is 'ha'.

    Returns
    -------
    par0 : list
        Default values of the 15-elements parameters.
    psig : list
        Prior deviations of the parameters 
    """

    if line.lower() == 'ha':
       par0qr = np.array([0.2, 0.5+3.5, -0.65, 0.5-2, 0., -0.26, np.log10(5.), np.log10(5.), -0.3, 0., -0.42, -0.42, -0.42, -0.87, 0.05])
       psigqr = np.array([0.5, 0.05, 0.1, 0.05, 0.2, 0.2, 0.1, 0.1, 1.4, 2.2, 0.05, 0.05, 0.02, 0.08, 1.e-4])
       par0ar = np.array([0.2, 0.50+3.5, -0.67, 0.50-2, 0.0, -0.21, np.log10(5.), np.log10(5.), -0.5, 0.1, -0.39, -0.39, -0.41, -0.74, 0.05])
       psigar = np.array([0.4, 0.05, 0.1, 0.05, 0.2, 0.2, 1.e-4, 1.e-4, 1.5, 1.8, 0.04, 0.04, 0.03, 0.10, 1.e-4])
    elif line.lower() == 'ca':
       par0qr = np.array([0., 0.4+4.6, -1.3, 1.4-2.9, 0.0, -0.55, np.log10(5.), np.log10(5.), 0., 0.9, -0.57, -0.57, -0.34, -1.16, 0.05])
       psigqr = np.array([0.5, 0.03, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 2.3, 1.4, 0.04, 0.06, 0.04, 0.24, 1.e-4])
       par0ar = np.array([0., 0.4+4.6, -1.3, 1.4-2.9, 0.02, -0.44, np.log10(5.), np.log10(5.), -0.5, 0.5, -0.57, -0.57, -0.30, -0.71, 0.05])
       psigar = np.array([0.5, 0.03, 0.1, 0.05, 0.1, 0.1, 0.01, 0.2, 2.2, 1.2, 0.04, 0.06, 0.04, 0.21, 0.001])
    
    par0 = (par0qr+par0ar)/2.
    psig = np.sqrt(0.5*(psigqr**2+(par0-par0qr)**2)+0.5*(psigar**2+(par0ar-par0)**2))*1.5 

    if line.lower() == 'ha':
          par0[10:12]= -0.50
          psig[10:12] = 0.07
          psig[12:14] = [0.05, 0.3]
          psig[8:10] = 3.0        
    elif line.lower() == 'ca':    
          par0[10:12]= -0.70
          psig[10:12] = 0.10
          psig[12:14] = [0.05, 0.3]
          psig[8:10] = 3.0
               
    return par0, psig

def absP_Voigt(wv1, wvc, w, a, line='ha'):
    """
    To determine absorption profile in the presence of damping. The profile 
    is normalzied for the peak value to be equal to 1.

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths from the line center in A.
    wvc : `float`
        central wavelength from the line center in A.
    w : `float`
        Doppler width in A.
    a : `float`
        dimensioness damping parameter.
    line : `str`, optional
        line designatio, either 'Ha' or 'Ca'. The default is 'ha'.

    Returns
    -------
    Phi : `numpy.ndarray`
        values of absorpsion coefficient (profile).
    """
    u = (wv1-wvc)/w   
    if line.lower() == 'ha':
        Phi = Voigt(u, a)/Voigt(0., a)
    if line.lower() == 'ca':
        dwv = np.array([.0857, .1426, .1696, .1952, .2433, .2871])-0.0857
        dA = 10**(np.array([6.33, 4.15, 3.47, 4.66, 1.94, 3.61])-6.33)
        # Phi = 0.  
        # for i in [0]:   
        #     Phi = Phi+Voigt(u-dwv[i]/w, a)*dA[i]
        Phi = Voigt(u-dwv[0]/w, a)*dA[0]
        Phi = Phi/Voigt(0.,a)
    return Phi

def absP_Gauss(wv1, wvc, w, line='ha'):
    """
    To determine Gaussian absorption profile normalized to have peak value of 1 

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths from line center in A.
    wvc : `float`
        central wavelength from line center in A.
    w : `float`
        Doppler width in A.
    line :`str`, optional
        line designation, either 'ha' or 'ca'. The default is 'ha'.

    Returns
    -------
    Phi : `numpy.ndarray`
        normalized absorption coefficent.

    """
    u = (wv1-wvc)/w   
    if line.lower() == 'ha':
         Phi = np.exp(-u**2)
    if line.lower() == 'ca':
        dwv = np.array([.0857, .1426, .1696, .1952, .2433,.2871])-0.0857
        dA = 10**(np.array([6.33, 4.15, 3.47, 4.66, 1.94, 3.61])-6.33)
        # Phi = 0.  
        # for i in [0]:
        #     Phi=Phi+np.exp(-(u-dwv[i]/w)**2)*dA[i]
        Phi = np.exp(-(u-dwv[0]/w)**2)*dA[0]
    return Phi

def _Sfromx(x, x0, a):
    return a[0]+a[1]*(x+x0)+ a[2]*(x+x0)**2

def cal_3layers(wv1, p, line='ha', phonly=False):
    """
    Calculate intensity profiles of a spectral line at three atmospheric levels.

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        Wavelengths measured from the line center.
    p : `numpy.ndarray`
        Array of 15 parameters.
    line : `str`, optional
        Line designation (default is 'ha').
    phonly : `bool`, optional
        If True, return only the intensity profile at the top of the photosphere.

    Returns
    -------
    I0, I1, I2 : `numpy.ndarray`
        Intensity profiles at the top of the chromosphere, middle of the chromosphere,
        and top of the photosphere, respectively.
    """
    ndim = p.ndim
    if ndim == 1:
        return cal_3layers1D(wv1, p, line=line, phonly=phonly)
    elif ndim == 2:
        return cal_3layers2D(wv1, p, line=line, phonly=phonly)

def cal_3layers1D(wv1, p, line='ha', phonly=False):
    """
    Calculate intensity profiles of a spectral line at three atmospheric levels.

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        Wavelengths measured from the line center.
    p : `numpy.ndarray`
        Array of 15 parameters.
    line : `str`, optional
        Line designation (default is 'ha').
    phonly : `bool`, optional
        If True, return only the intensity profile at the top of the photosphere.

    Returns
    -------
    I0, I1, I2 : `numpy.ndarray`
        Intensity profiles at the top of the chromosphere, middle of the chromosphere,
        and top of the photosphere, respectively.
    """

    
    #  Change of Variables    
    c = const.c.value * 1e-3
    wvline = get_centerWV(line)
    wvp = p[0] / c * wvline
    eta, wp, ap, Sp, S2 = 10 ** p[1:6]
    tau2, tau1 = 10 ** p[6:8]
    wv01, wv00 = p[8:10] / c * wvline
    w1, w0 = 10 ** p[10:12]
    S1, S0 = 10 ** p[12:14]
    wg = p[14]
    wv02 = wvp
    w2 = wp

    # Photosphereic Contribution
    rlamb = eta * absP_Voigt(wv1, wvp, wp, ap, line=line) + 1
    I2 = S2 + (Sp - S2) / rlamb

    if phonly:
        return I2
       
    xvalues = np.array([-0.774597, 0, 0.774597])
    weights = np.array([0.55556, 0.888889, 0.555556])
 
    # Lower Chromosphere
    xx_grid = (xvalues[:, None] + 1) / 2
    wvcenter_lc = wv01 + (wv02 - wv01) * xx_grid
    width_lc = w1 + (w2 - w1) * xx_grid
    a_lc = (wg * xx_grid) / width_lc

    dummy_lc = np.sum(weights[:, None] * absP_Voigt(wv1, wvcenter_lc, width_lc, a_lc, line=line), axis=0)
    taulamb_lc = tau2 / 2 * dummy_lc

    A = [S0, -1.5 * S0 + 2 * S1 - 0.5 * S2, 0.5 * S0 - S1 + 0.5 * S2]

    x_grid_outer = (xvalues[:, None] + 1) / 2
    x_grid_inner = (xvalues[:, None, None] + 1) * x_grid_outer / 2 

    wvcenter_inner = wv01 + (wv02 - wv01) * x_grid_inner
    width_inner = w1 + (w2 - w1) * x_grid_inner
    a_inner = (wg * x_grid_inner) / width_inner

    dummy_inner = np.sum(weights[:, None, None] * absP_Voigt(wv1, wvcenter_inner, width_inner, a_inner, line=line), axis=0)
    
    tlamb_inner = tau2 * x_grid_outer / 2 * dummy_inner

    wvcenter_outer = wv01 + (wv02 - wv01) * x_grid_outer
    width_outer = w1 + (w2 - w1) * x_grid_outer
    a_outer = (wg * x_grid_outer) / width_outer

    S = _Sfromx(x_grid_outer, 1., A)

    integral_lc = np.sum(weights[:, None] * S * np.exp(-tlamb_inner) * absP_Voigt(wv1, wvcenter_outer, width_outer, a_outer, line=line), axis=0)

    I1 = I2 * np.exp(-taulamb_lc) + tau2 / 2 * integral_lc

    # Upper Chromosphere         
    wvcenter_uc = wv00 + (wv01 - wv00) * xx_grid
    width_uc = w0 + (w1 - w0) * xx_grid

    dummy_uc = np.sum(weights[:, None] * absP_Gauss(wv1, wvcenter_uc, width_uc, line=line), axis=0)
    taulamb_uc = tau1 / 2 * dummy_uc

    wvcenter_inner_uc = wv00 + (wv01 - wv00) * x_grid_inner
    width_inner_uc = w0 + (w1 - w0) * x_grid_inner

    dummy_inner_uc = np.sum(weights[:, None, None] * absP_Gauss(wv1, wvcenter_inner_uc, width_inner_uc, line=line), axis=0)
    tlamb_inner_uc = tau1 * x_grid_outer / 2 * dummy_inner_uc

    wvcenter_outer_uc = wv00 + (wv01 - wv00) * x_grid_outer
    width_outer_uc = w0 + (w1 - w0) * x_grid_outer

    S_uc = _Sfromx(x_grid_outer, 0., A)

    integral_uc = np.sum(weights[:, None] * S_uc * np.exp(-tlamb_inner_uc) * absP_Gauss(wv1, wvcenter_outer_uc, width_outer_uc, line=line), axis=0)

    I0 = I1 * np.exp(-taulamb_uc) + tau1 / 2 * integral_uc
      
    return I0, I1, I2

def cal_3layers2D(wv1, p, line='ha', phonly=False):
    """
    Calculate intensity profiles of a spectral line at three atmospheric levels.

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        Wavelengths measured from the line center.
    p : `numpy.ndarray`
        Array of 15 parameters.
    line : `str`, optional
        Line designation (default is 'ha').
    phonly : `bool`, optional
        If True, return only the intensity profile at the top of the photosphere.

    Returns
    -------
    I0, I1, I2 : `numpy.ndarray`
        Intensity profiles at the top of the chromosphere, middle of the chromosphere,
        and top of the photosphere, respectively.
    """

    
    #  Change of Variables    
    c = const.c.value * 1e-3
    wvline = get_centerWV(line)
    wvp = p[0] / c * wvline
    eta, wp, ap, Sp, S2 = 10 ** p[1:6]
    tau2, tau1 = 10 ** p[6:8]
    wv01, wv00 = p[8:10] / c * wvline
    w1, w0 = 10 ** p[10:12]
    S1, S0 = 10 ** p[12:14]
    wg = p[14]
    wvp = wvp[:, None]
    eta = eta[:, None]
    wp = wp[:, None]
    ap = ap[:, None]
    Sp = Sp[:, None]
    S2 = S2[:, None]
    tau2 = tau2[:, None]
    tau1 = tau1[:, None]
    wv01 = wv01[:, None]
    wv00 = wv00[:, None]
    w1 = w1[:, None]
    w0 = w0[:, None]
    S1 = S1[:, None]
    S0 = S0[:, None]
    wg = wg[:, None]
    wv02 = wvp
    w2 = wp

    # Photosphereic Contribution
    rlamb = eta * absP_Voigt(wv1, wvp, wp, ap, line=line) + 1
    I2 = S2 + (Sp - S2) / rlamb

    if phonly:
        return I2
       
    xvalues = np.array([-0.774597, 0, 0.774597])
    weights = np.array([0.55556, 0.888889, 0.555556])
 
    # Lower Chromosphere
    xx_grid = (xvalues[:, None, None] + 1) / 2
    wvcenter_lc = wv01 + (wv02 - wv01) * xx_grid
    width_lc = w1 + (w2 - w1) * xx_grid
    a_lc = (wg * xx_grid) / width_lc

    dummy_lc = np.sum(weights[:, None, None] * absP_Voigt(wv1, wvcenter_lc, width_lc, a_lc, line=line), axis=0)
    taulamb_lc = tau2 / 2 * dummy_lc

    A = [S0, -1.5 * S0 + 2 * S1 - 0.5 * S2, 0.5 * S0 - S1 + 0.5 * S2]

    x_grid_outer = (xvalues[:, None, None] + 1) / 2
    x_grid_inner = (xvalues[:, None, None, None] + 1) * x_grid_outer / 2 

    wvcenter_inner = wv01 + (wv02 - wv01) * x_grid_inner
    width_inner = w1 + (w2 - w1) * x_grid_inner
    a_inner = (wg * x_grid_inner) / width_inner

    dummy_inner = np.sum(weights[:, None, None, None] * absP_Voigt(wv1, wvcenter_inner, width_inner, a_inner, line=line), axis=0)
    
    tlamb_inner = tau2 * x_grid_outer / 2 * dummy_inner

    wvcenter_outer = wv01 + (wv02 - wv01) * x_grid_outer
    width_outer = w1 + (w2 - w1) * x_grid_outer
    a_outer = (wg * x_grid_outer) / width_outer

    S = _Sfromx(x_grid_outer, 1., A)

    integral_lc = np.sum(weights[:, None, None] * S * np.exp(-tlamb_inner) * absP_Voigt(wv1, wvcenter_outer, width_outer, a_outer, line=line), axis=0)

    I1 = I2 * np.exp(-taulamb_lc) + tau2 / 2 * integral_lc

    # Upper Chromosphere
    wvcenter_uc = wv00 + (wv01 - wv00) * xx_grid
    width_uc = w0 + (w1 - w0) * xx_grid

    dummy_uc = np.sum(weights[:, None, None] * absP_Gauss(wv1, wvcenter_uc, width_uc, line=line), axis=0)
    taulamb_uc = tau1 / 2 * dummy_uc

    wvcenter_inner_uc = wv00 + (wv01 - wv00) * x_grid_inner
    width_inner_uc = w0 + (w1 - w0) * x_grid_inner

    dummy_inner_uc = np.sum(weights[:, None, None, None] * absP_Gauss(wv1, wvcenter_inner_uc, width_inner_uc, line=line), axis=0)
    tlamb_inner_uc = tau1 * x_grid_outer / 2 * dummy_inner_uc

    wvcenter_outer_uc = wv00 + (wv01 - wv00) * x_grid_outer
    width_outer_uc = w0 + (w1 - w0) * x_grid_outer

    S_uc = _Sfromx(x_grid_outer, 0., A)

    integral_uc = np.sum(weights[:, None, None] * S_uc * np.exp(-tlamb_inner_uc) * absP_Gauss(wv1, wvcenter_outer_uc, width_outer_uc, line=line), axis=0)

    I0 = I1 * np.exp(-taulamb_uc) + tau1 / 2 * integral_uc
      
    return I0, I1, I2


def cal_residue(pf, wv1, intensity,  p0, psig, p, line, free, constr):
    """
     To calculate the residual array to be used  for fitting based on least_squares

    Parameters
    ----------
    pf : `numpy.ndarray`
        free parameters to be determined.
    wv1 : `numpy.ndarray`
        wavelengths measured from line center .
    intensity : `numpy.ndarray`
        intensities.
    p0 : `numpy.ndarray`
        default model parameters .
    psig : `numpy.ndarray`
        prior deviation of model parameters .
    p : `numpy.ndarray`
        model parameters .
    line : `str`
        line designation.
    free  : `int`
         indexes of free parameter
    constr : `int`
         indexes of constrained parameter      
    Returns
    -------
    Res : `numpy.ndarray`
        residuals.

    """
   # free, constr = ThreeLayerParControl(line=line)
    p[free]=pf
    model = cal_3layers(wv1,p, line=line)[0]

    sig = get_Inoise(intensity, line=line)    
    resD = (intensity-model)/sig 
    
    resC = (p[constr]-p0[constr])/psig[constr]
    resC = np.append(resC, (p[10]-p[11])/np.sqrt(psig[10]**2+psig[11]**2))
    resC = np.append(resC, (p[8]-p[9])/np.sqrt(psig[8]**2+psig[9]**2))
    
    logS1, logS0 = p[12:14]
    logSp, logS2 = p[4:6]
    logw1, logw0 = p[10:12]
    
    if line.lower() == 'ha': 
        resC = np.append(resC, Penalty((logw1-logw0)/0.03) )
        resC = np.append(resC, Penalty((logS2-logS1)/0.03) )
        resC = np.append(resC, Penalty((logS1-logS0)/0.03) )
        resC = np.append(resC, Penalty((logS0+1.5)/0.03) )

    if line.lower() == 'ca':
        resC = np.append(resC, Penalty((logw1-logw0)/0.03) )
        resC = np.append(resC, Penalty((logS1-logS2)/0.03) )
        resC = np.append(resC, Penalty((logS1-logS0)/0.03) )
        resC = np.append(resC, Penalty((logS0+1.5)/0.03) )
      
    Res = np.append(resD/np.sqrt(len(resD)), resC/np.sqrt(len(resC)))
    
    return Res 

def par0_3layers(wv1, prof,  line='Ha'):
    """
    Determine the initial guess of model parameters  and  their prior deviations

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths measured from line center.
    prof : `numpy.ndarray`
        intesity profile.
    line : `str`, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    par : `numpy.ndarray`
         model parameters
        
    psig : `numpy.ndarray`   
        prior deviations of model parameters
    

    """
    # Preparation
    c = const.c.value * 1e-3
    par0, psig0 = parDefault(line=line) 
    par = par0.copy()
    psig = psig0.copy()
    profav = savgol_filter(prof, window_length=7, polyorder=2)   
    wvline = get_centerWV(line)
    pure = get_pure(wv1+wvline, line=line)
    sel = get_sel(wv1, line)  

    fprofav = interp1d(wv1[pure], profav[pure])  
    
    wvp, dwv= get_photoLineWV (line, wv1.min()+wvline, wv1.max()+wvline)
    pline = abs(wv1+wvline-wvp) <= 2*dwv    
    dwc = get_Linecenter(wv1[pline]+wvline-wvp, profav[pline], nd=2)
    vp = dwc/wvp*c
    par[0] = vp  
 
# Photospheric Layer
        
    eta = 10**par[1]
    ap = 10**par[3]
    if line.lower() =='ha': wv1a, wv1b = 1., 4.
    else: wv1a, wv1b = 0.7, 5.
    #wv1ap = wv1a+vp/3.e5*wvline
#    intap = fprofav(wv1a)
   # wv1an = -wv1a+vp/3.e5*wvline    
#    intan = fprofav(-wv1a)
    inta =  (fprofav(wv1a)+fprofav(-wv1a))/2.
    intb = (fprofav(wv1b)+fprofav(-wv1b))/2.
    # print(inta, intb, T)
   
    Tav = Trad(inta,line)
    wha, wca = Dwidth(Tav, 1.)
   
    if line.lower() == 'Ha':
        wp = wha       
    else:
        wp = wca
    par[2] = np.log10(wp)

    eta  = 10**par[1]    
    qa = 1./(eta*Voigt(wv1a/wp, ap)/Voigt(0.,ap)+1. )
    qb = 1./(eta*Voigt(wv1b/wp, ap)/Voigt(0.,ap)+1. )
    d1 = (inta-intb)/(qa-qb)
    d0 = (inta*qb-intb*qa)/(qb-qa)
    S2 = d0
    Sp = d1 + S2
    par[4] = np.log10(np.maximum(Sp, intb*0.5))
    par[5] = np.log10(np.maximum(S2, intb*0.02))
    if line.lower() == 'Ha': 
        par[14] = wp*1.E-3 #ap*0.0
        par[12] = par[5]*0.5 + np.log10(profav[sel].min())*0.5  
    else: 
        par[14] = wp*1.E-3 #ap*0.0
        par[12] = par[5]+0.1 #np.log10(profav[sel].max())*0.6 + np.log10(profav[sel].min())*0.4 
    S1=10**par[12]
    int0 = profav[sel].min()
    x1=1./10**par[7]
    S0 =  np.maximum((int0-x1*(2*S1-0.5*S2))/(1-1.5*x1), 10.**-2.)    
    par[13]=np.log10(S0)  
    modelph = cal_3layers(wv1[sel],par, line=line, phonly=True)       
    weight =np.maximum(modelph*1.-profav[sel], 0.)*(abs(wv1[sel])< 0.5)          
    wv0 =( weight*wv1[sel]).sum()/np.maximum(weight.sum(), 1.)
    vc = wv0/wvline*3.e5
    par[9] = vc
    par[8] = vc
            
    return par, psig

def par0_3layers_2D(wv1, prof,  line='Ha'):
    """
    Determine the initial guess of model parameters  and their prior deviations

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths measured from line center.
    prof : `numpy.ndarray`
        intesity profile.
    line : str, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    par : `numpy.ndarray`
         model parameters
        
    psig : `numpy.ndarray`   
        prior deviations of model parameters
    """
    # Preparation
    c = const.c.value * 1e-3
    par0, psig0 = parDefault(line=line) 
    par = par0.copy()[:,None] + np.zeros((len(par0), len(prof)))
    psig = psig0.copy()[:,None] + np.zeros((len(par0), len(prof)))
    profav = savgol_filter(prof, window_length=7, polyorder=2, axis=-1)   
    wvline = get_centerWV(line)
    pure = get_pure(wv1+wvline, line=line)
    sel = get_sel(wv1, line)
    profi = profav[...,pure]  
    ww = wv1[pure]

    tfprofav = interp1d(ww, profi, axis=1)

    wvp, dwv = get_photoLineWV(line, wv1.min()+wvline, wv1.max()+wvline)
    pline = abs(wv1+wvline-wvp) <= 2*dwv    
    dwc = get_Linecenter(wv1[pline]+wvline-wvp, profav[:,pline], nd=2)
    vp = dwc/wvp*c
    par[0] = vp
 
# Photospheric Layer
        
    eta = 10**par[1]
    ap = 10**par[3]
    if line.lower() =='ha': wv1a, wv1b = 1., 4.
    else: wv1a, wv1b = 0.7, 5.
    wvarr = np.array([wv1a, -wv1a, wv1b, -wv1b])
    Iarr = tfprofav(wvarr)
    inta = (Iarr[:,0]+Iarr[:,1])/2.
    intb = (Iarr[:,2]+Iarr[:,3])/2.
   
    Tav = Trad(inta,line)
    wha, wca = Dwidth(Tav, 1.)
   
    if line.lower() == 'ha':
        wp = wha       
    else:
        wp = wca
    par[2] = np.log10(wp)

    eta  = 10**par[1]    
    qa = 1./(eta*Voigt(wv1a/wp, ap)/Voigt(0.,ap)+1. )
    qb = 1./(eta*Voigt(wv1b/wp, ap)/Voigt(0.,ap)+1. )
    d1 = (inta-intb)/(qa-qb)
    d0 = (inta*qb-intb*qa)/(qb-qa)
    S2 = d0
    Sp = d1 + S2
    par[4] = np.log10(np.maximum(Sp, intb*0.5))
    par[5] = np.log10(np.maximum(S2, intb*0.02))
    if line.lower() == 'ha': 
        par[14]   = wp*1.E-3 #ap*0.0
        par[12] =   par[5]*0.5 + np.log10(profav[:,sel].min(-1))*0.5  
    else: 
        par[14]   = wp*1.E-3 #ap*0.0
        par[12] =  par[5]+0.1 #np.log10(profav[sel].max())*0.6 + np.log10(profav[sel].min())*0.4 
    S1 = 10**par[12]
    int0 = profav[:,sel].min(-1)
    x1 = 1./10**par[7]
    S0 =  np.maximum((int0 - x1 * (2 * S1 - 0.5 * S2)) / (1 - 1.5*x1), 10.**-2.)    
    par[13] = np.log10(S0)
    modelph = cal_3layers(wv1[sel], par, line=line, phonly=True)
    weight = np.maximum(modelph * 1. - profav[:,sel], 0.)*(abs(wv1[sel])< 0.5)
    wv0 = (weight * wv1[sel]).sum(-1)/np.maximum(weight.sum(-1), 1.)
    vc = wv0 / wvline * c
    par[9] = vc
    par[8] = vc
            
    return par, psig

def ParControl(line='ha'):
    """
    To provide the indexes of the free model parameters and 
    the priorly constrained model parameters

    Parameters
    ----------
    line : `str`, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    free : `numpy.ndarray`
        indexes of free parameters to be determined from the fitting.  
    constr : `numpy.ndarray`
        indexes of priorly constrained parameters. 

    """
    if line.lower() == 'ha':
        free   = [1,2,3,4,5,8,9,10,11,12,13]
        constr = [1,2,3,8,9,10,11,12,13]
       
    else:    
        free   = [1,2,3,4,5,8,9,10,11,12,13]
        constr = [1,2,3, 8,9,10,11,12,13]
    return free, constr

def Model(wv1, prof, **kwargs):
    """
    Do the three-layer spectrla inversion of a line profile

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths (in Angstrom) measured from line center.
    prof : `numpy.ndarray`
        profile of intensities (normalized by the disk center continuum intensity).
    sel : `numpy.ndarray`, optional
        True for the indexes of data to be used for fitting. The default is None.
    par : `numpy.ndarray`, optional
         initial estimates of the model parameters. The default is None.
    par0 : `numpy.ndarray`, optional
        default model parameters. The default is None.
    psig : `numpy.ndarray`, optional
        prior deviations of model parameters. The default is None.
    free : `numpy.ndarray`, optional
        indexes of free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    constr : `numpy.ndarray`, optional
        indexes of constrained free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    line : `str`, optional
        line designation The default is 'ha'.

    Returns
    -------
    par : `numpy.ndarray`
        model parameters.
    I0 : `numpy.ndarray`
        intensity profile at the top of the chromosphere    
    I1 : `numpy.ndarray`
        intensity profile in the middle of the chromosphere
    I2 : `numpy.ndarray`
        intensity profile at the top of the photosphere
    epsD : `float`
        error of data fitting.
    epsP : `float`
        deviation of parameters from prior conditions.

    """
    ndim = prof.ndim
    if ndim == 1:
        return _Model_1D(wv1, prof, **kwargs)
    else:
        return _Model_ND(wv1, prof, **kwargs)

def _Model_1D(wv1, prof,  sel=None, par=None, par0=None, psig=None,  free=None, constr=None, line='ha', ncore=None):
    """
    Do the three-layer spectrla inversion of a line profile

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths (in Angstrom) measured from line center.
    prof : `numpy.ndarray`
        profile of intensities (normalized by the disk center continuum intensity).
    sel : `numpy.ndarray`, optional
        True for the indexes of data to be used for fitting. The default is None.
    par : `numpy.ndarray`, optional
         initial estimates of the model parameters. The default is None.
    par0 : `numpy.ndarray`, optional
        default model parameters. The default is None.
    psig : `numpy.ndarray`, optional
        prior deviations of model parameters. The default is None.
    free : `numpy.ndarray`, optional
        indexes of free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    constr : `numpy.ndarray`, optional
        indexes of constrained free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    line : `str`, optional
        line designation The default is 'ha'.

    Returns
    -------
    par : `numpy.ndarray`
        model parameters.
    I0 : `numpy.ndarray`
        intensity profile at the top of the chromosphere    
    I1 : `numpy.ndarray`
        intensity profile in the middle of the chromosphere
    I2 : `numpy.ndarray`
        intensity profile at the top of the photosphere
    epsD : `float`
        error of data fitting.
    epsP : `float`
        deviation of parameters from prior conditions.

    """

    if prof.min() >= 0.9*prof.max() or prof.max() < 0.05 :
       pari, psig = parDefault( line=line)
       par = pari
       I2 = prof
       I0 = prof
       I1 = prof
       epsD=-1.
       epsP =-1.
       return   (par,  I0, I2, I1, epsD, epsP) 
    if par0 is None:
        pari, psig = par0_3layers(wv1, prof, line=line) 
    else:
        pari = par0.copy()
    
    if free is None: 
        free, constr0 = ParControl(line=line)
    if constr is None:
        free0, constr = ParControl(line=line)
      
    if sel is None: sel = get_sel(wv1, line)
  
    if par is None: par = np.copy(pari)
    parf=par[free]
    
    res_lsq= least_squares(cal_residue, parf,
                           args=(wv1[sel], prof[sel], pari, psig,  par, line, free, constr),
                           jac='2-point',  max_nfev=100, ftol=1e-3) 
    
    
    parf = np.copy(res_lsq.x)
    par[free] = parf
    residual= res_lsq.fun

    Ndata = len(wv1[sel]) 
    epsD = np.sqrt((residual[0:Ndata]**2).sum())
    epsP = np.sqrt((residual[Ndata:]**2).sum())
         
    I0, I1, I2 = cal_3layers(wv1,par, line=line)
    

    return  (par,  I0, I1, I2,  epsD, epsP)

def _Model_ND(wv1, prof,  sel=None, par=None, par0=None, psig=None,  free=None, constr=None, line='ha', ncore=-1):
    """
    Do the three-layer spectrla inversion of a line profile

    Parameters
    ----------
    wv1 : `numpy.ndarray`
        wavelengths (in Angstrom) measured from line center.
    prof : `numpy.ndarray`
        profile of intensities (normalized by the disk center continuum intensity).
    sel : `numpy.ndarray`, optional
        True for the indexes of data to be used for fitting. The default is None.
    par : `numpy.ndarray`, optional
         initial estimates of the model parameters. The default is None.
    par0 : `numpy.ndarray`, optional
        default model parameters. The default is None.
    psig : `numpy.ndarray`, optional
        prior deviations of model parameters. The default is None.
    free : `numpy.ndarray`, optional
        indexes of free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    constr : `numpy.ndarray`, optional
        indexes of constrained free parameters to be determined. The default is None. 
        If None, the indexes are determined by the function ThreeLayerParControl().
    line : `str`, optional
        line designation The default is 'ha'.
    ncore: `int`, optional
        Number of CPU cores. The default is -1 (use all cores)

    Returns
    -------
    par : `numpy.ndarray`
        model parameters.
    I0 : `numpy.ndarray`
        intensity profile at the top of the chromosphere    
    I1 : `numpy.ndarray`
        intensity profile in the middle of the chromosphere
    I2 : `numpy.ndarray`
        intensity profile at the top of the photosphere
    epsD : `float`
        error of data fitting.
    epsP : `float`
        deviation of parameters from prior conditions.

    """
    sh = prof.shape
    pm = prof.min(-1)
    pM = prof.max(-1)
    wh = np.logical_or(pm >= .9*pM,  pM < .05)
    pard, dummy = parDefault(line=line)
    I2 = np.zeros(sh)
    I1 = np.zeros(sh)
    I0 = np.zeros(sh)
    epsD = np.zeros(sh[:-1])
    epsP = np.zeros(sh[:-1])
    epsD[wh] = -1.
    epsP[wh] = -1.
    if par0 is None:
        pari, psigi = par0_3layers_2D(wv1, prof[~wh], line=line)
    else:
        pari = par0.copy()
        psigi = psig.copy()
    if par is None:
        parsh = [len(pard)]+list(sh[:-1])
        par = np.zeros(parsh)
        par[:,~wh] = np.copy(pari)
    par[:,wh] = par0
    I2[wh] = prof[wh]
    I1[wh] = prof[wh]
    I0[wh] = prof[wh]

    
    if free is None: 
        free, constr0 = ParControl(line=line)
    if constr is None:
        free0, constr = ParControl(line=line)
    nfree = len(free)
    if sel is None: sel = get_sel(wv1, line)
  
    ww = wv1[sel]
    parf = par[free]
    parf = parf[:,~wh]
    fp = prof[~wh][:,sel]
    residual = np.zeros((fp.shape[0],sel.sum()+len(par)))
    pi = par[:,~wh]
    

    
    if 0:
        for i in range(fp.shape[0]):
            res = lsq_single(parf[:,i], ww, fp[i], pari[:,i], psigi[:,i],  pi[:,i], line, free, constr)
            parf[:,i] = res[:nfree]
            residual[i] = res[nfree:]
    if 1:
        res = Parallel(n_jobs=ncore)(delayed(lsq_single)(parf[:,i], ww, fp[i], pari[:,i], psigi[:,i],  pi[:,i], line, free, constr) for i in range(fp.shape[0]))
        res = np.array(res)
        parf = res[:,:nfree].T
        residual = res[:,nfree:]

    for i, f in enumerate(free):
        par[f, ~wh] = parf[i]

    Ndata = len(wv1[sel])
    depsD = np.sqrt((residual[:, 0:Ndata]**2).sum(-1))
    depsP = np.sqrt((residual[:, Ndata:]**2).sum(-1))
    epsD[~wh] = depsD
    epsP[~wh] = depsP
         
    I0[~wh], I1[~wh], I2[~wh] = cal_3layers(wv1, par[:,~wh], line=line)
    

    return  (par,  I0, I1, I2,  epsD, epsP)

def RadLoss(p, line='ha'):
    """
    To calculate the radiative losses of the upper chromosphere and
    the lower chromosphere, respectively, from the three-layer model parameters

    Parameters
    ----------
    p : `numpy.ndarray`
        parameters of three layer models.
    line : `str`, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    Radloss01 : `float` or `numpy.ndarray`
        radiative loss from the upper chromosphere in unit of kW/m^2
        
    Radloss12 : `float` or `numpy.ndarray`
        raditive loss from the lower chromosphere in unit of kW/m^2   
    """
    nd = p.ndim
    if nd == 1:
        return _RadLossBase(p, line)
    elif nd == 2:
        return _RadLoss2D(p, line)
    else:
        osh = p.shape
        sh = (osh[0], np.prod(osh[1:]))
        p2 = p.reshape(sh)
        RL = _RadLoss2D(p2, line)
        return RL[0].reshape(osh[1:]), RL[1].reshape(osh[1:])

def _RadLossBase(p, line='ha'):
    """
    To calculate the radiative losses of the upper chromosphere and
    the lower chromosphere, respectively, from the three-layer model parameters

    Parameters
    ----------
    p : `numpy.ndarray`
        parameters of three layer models.
    line : `str`, optional
        line designation. The default is 'Ha'.

    Returns
    -------
    Radloss01 : `float`
        radiative loss from the upper chromosphere in unit of kW/m^2
        
    Radloss12 : `float`
        rdaitive loss from the lower chromosphere in unit of kW/m^2   

    """

    c = const.c.value * 1e-3
    xvalues5p = np.array([-0.9062, -0.5385, 0., 0.5385, 0.9062])
    weights5p = np.array([ 0.2369, 0.4786, 0.5689, 0.4786, 0.2369])
    xvalues3p = np.array([-0.774597, 0, 0.774597])
    weights3p = np.array([0.55556, 0.888889, 0.555556])
    
#    if line =='Ha':
    lambdaMax = 6.
    tauM = 100.

    wvline = get_centerWV(line)
    wvp = p[0] / c * wvline
    eta, wp, ap, Sp, S2 = 10. ** p[1:6]
    
    tau2, tau1 = 10 ** p[6:8]    
    wv01, wv00 = p[8:10] / c * wvline
    w1, w0 = 10 ** p[10:12]
    S1, S0 = 10 ** p[12:14]
    wg = p[14]
    wv02 = wvp
    w2 = wp

    def gf(wv1, x, level):
        if level == 1:
            value = absP_Gauss(wv1, wv00+(wv01-wv00)*x, w0 + (w1-w0)*x, line=line)
        elif level == 2:
            wvcenter= (wv01+(wv02-wv01) * x)
            width = (w1 + (w2-w1) * x)
            a = (wg * x) / width
            value = absP_Voigt(wv1, wvcenter, width, a, line=line)
        elif level == 3:
            value = (eta * absP_Voigt(wv1, wvp, wp, ap, line=line)+1.)
        return value

    def dff(wv1, x1, x2, level):
         if level == 3:
             value = (x2-x1) * gf(wv1, x1, level)
         else:    
             xx = (xvalues3p * (x2-x1)/2.) + (x2+x1)/2.
             wdx = weights3p * (x2-x1)/2.
             value = 0.
             for i in range(len(xx)):
                 value = value + gf(wv1, xx[i], level) * wdx[i]
         return value
    a0 = S0
    a1 = (-1.5*S0+2*S1-0.5*S2)
    a2 = (0.5*S0-S1+0.5*S2)

    def Sf(x, level):
       if level == 1:
           x0 = 0.
           value = a0+a1*(x+x0)+ a2*(x+x0)**2
       elif level == 2:
           x0 = 1.
           value = a0+a1*(x+x0)+ a2*(x+x0)**2
       elif level == 3:
           value = S2 + (Sp-S2)*(tauM*x)
       return value
#        
            
    
    xs = (xvalues5p + 1) * 0.5
    weightdx = weights5p * 0.5
    Nx = len(xs)
 
#   Applyng Simpson's 3/8 rule making use of Cubic spline
    
    Nlambda = int(10 * lambdaMax)
    Nlambda = Nlambda + (3 - ((Nlambda-1) % 3))
    index = np.arange(Nlambda)
    
    w = np.ones(Nlambda, float) * 2
    s = index % 3 == 0
    w[s] = 3.
    w[0] = 1.
    w[-1] = 1.
    lambdas = (index / (Nlambda - 1) - 0.5) * lambdaMax
    weightdlambda = (3./8.) * (lambdas[-1] - lambdas[0]) / (Nlambda - 1) * w
        
    Nlambda = len(lambdas)
    
    
    Nl = 2
    
    gs = np.zeros((Nlambda, Nx, Nl))
    fs = np.zeros((Nlambda, Nx, Nl))
    
    
    for l in range(Nl):
        for j in range(Nx):
            gs[:,j,l] = gf(lambdas, xs[j], l+1)
            if j == 0:
                fs[:,j,l] = dff(lambdas, 0., xs[j], l+1)
            else:
                fs[:,j,l] = fs[:, j-1,l]+dff(lambdas, xs[j-1], xs[j], l+1)
    taulambda1 =   tau1*( fs[:, -1, 0]+dff(lambdas, xs[-1], 1., 1))
    taulambda2 =   tau2*( fs[:, -1, 1]+dff(lambdas, xs[-1], 1., 2))

    K12 = np.zeros((Nx, Nl))
    K01 = np.zeros((Nx, Nl))
    S = np.zeros((Nx, Nl))
 
    
    for j in range(Nx):            
            tmp = -expn(2, taulambda1-tau1*fs[:,j,0]) + expn(2, taulambda1+taulambda2-tau1*fs[:,j,0])
            tmp = tau1*tmp*gs[:,j,0]
            K12[j,0] = (weightdlambda*tmp).sum()
            tmp = expn(2,tau1*fs[:,j,1])+ expn(2, taulambda2-tau2*fs[:,j,1])
            tmp = tau2*tmp*gs[:,j,1]
            K12[j,1] = (weightdlambda*tmp).sum()
            tmp = expn(2, tau1*fs[:,j,0])+ expn(2, taulambda1-tau1*fs[:,j,0])
            tmp = tau1*tmp*gs[:,j,0]
            K01[j,0] = (weightdlambda*tmp).sum()
            tmp = expn(2,taulambda1+ tau2*fs[:,j,1])- expn(2, tau2*fs[:,j,1])
            tmp = tau2*tmp*gs[:,j,1]
            K01[j,1] = (weightdlambda*tmp).sum()
            if Nl > 2:
                tmp = -expn(2,tauM*fs[:,j,2])+expn(2,  taulambda2+tauM*fs[:,j,2])
                tmp = tauM*tmp*gs[:,j,2]
                K12[j,2] =(weightdlambda*tmp).sum()        
                tmp = -expn(2,taulambda2+tauM*fs[:,j,2])+expn(2, taulambda1+ taulambda2+tauM*fs[:,j,2])
                tmp =tauM* tmp*gs[:,j,2]
                K01[j,2] = (weightdlambda*tmp).sum()
        
            for l in range(Nl):
                S[j,l] = Sf(xs[j], l+1)       
    
    Radloss01, Radloss12 = 0., 0
    for l in range(Nl):
        Radloss01 += 2*np.pi * (weightdx*S[:,l]*K01[:,l]).sum()
        Radloss12 += 2*np.pi * (weightdx*S[:,l]*K12[:,l]).sum()
    if line.lower() == 'ha':
       Radloss01 *= 2.84  # in kW/m^2
       Radloss12 *= 2.84
    elif line.lower() == 'ca':
       Radloss01 *= 1.76  
       Radloss12 *= 1.76  # in kW/m^2
      
    return Radloss01, Radloss12

def _RadLoss2D(p, line='ha'):
    psh = p.shape[1]
    RL1 = np.zeros(psh)
    RL2 = np.zeros(psh)
    for i in prange(psh):
        RL1[i], RL2[i] = _RadLossBase(p[:,i], line)
    return RL1, RL2
        

def lsq_single(iparf, iww, ifp, ipari, ipsigi, ipi, iline, ifree, iconstr):
    res_lsq = least_squares(cal_residue, iparf,
                            args=(iww, ifp, ipari, ipsigi,  ipi, iline, ifree, iconstr), jac='2-point',  max_nfev=100, ftol=1e-3)
    return np.append(res_lsq.x, res_lsq.fun)
    
def testMLSI(Infile):
    fiss = FISS(Infile)
    fiss.correction()

def testMLSI_4(Infile):
    fiss = FISS(Infile)
    fiss.correction()

