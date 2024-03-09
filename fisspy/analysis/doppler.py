"""
Doppler

This module calculate line of sight doppler velocities for
each pixels of a FISS fts data.
"""
from __future__ import absolute_import, division

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"

import numpy as np
from interpolation.splines import LinearSpline, CubicSpline
from astropy.constants import c
from scipy.signal import fftconvolve as conv
from fisspy.correction.correction import get_InstShift
from scipy.ndimage import gaussian_filter1d

__all__ = ['lambdameter', 'LOS_velocity']


def lambdameter(wv, data0, hw=0.05, iwc=None, corInstShift=False, refSpec=None, wvRange=None, method='linear', cm='kang'):
    """
    Determine the Lambdameter chord center for a given half width or intensity.

    Parameters
    ----------
    wv: `~numpy.ndarray`
        A Calibrated wavelength.
    data: `~numpy.ndarray`
        n (n=2 or n=3) dimensional spectral profile data,
        the last dimension component must be the spectral component,
        and the size is equal to the size of wv.
    hw: `float`, optional
        A half width of the horizontal line segment.
    corInstShift: `bool`, optional
        If True, correct the instrument shift.
        Default is False
    refSpec: `float`, optional
    wvRange: `list`, optional

    Returns
    -------
    wc : `~numpy.ndarray`
        n dimensional array of central wavelength values.
    intc : `~numpy.ndarray`
        n dimensional array of intensies of the line segment.

    Notes
    -----
        This function is based on the IDL code BISECTOR_D.PRO
        written by J. Chae.
        This function was dratsically modified  by J. Chae in 2023.
        This function was optimized by J. Kang in Mar 2024.

    Example
    -------
    >>> from fisspy.analysis import doppler
    >>> wc, inten = doppler.labdameter(wv,data,0.2)
    """

    if wvRange is None:
        wvRange = [wv[0], wv[-1]]
    ss = (wv >= min(wvRange)) * (wv <= max(wvRange))


    shape = data0.shape
    nw = shape[-1]
    reshape = shape[:-1]
    dkern = np.array([[-1, 1, 0, 1, -1]])
    ndim = data0.ndim
    dwv = wv[1]-wv[0]

    if ndim >=4:
        raise ValueError('The dimension of data0 must be 2 or 3.')
    if wv.shape[0] != nw:
        raise ValueError('The number of elements of wv and '
        'the number of elements of last axis for data are not equal.')
    if hw <= 0:
        raise ValueError('The half-width value must be greater than 0.')
    
    # correct instrumental shift by the seeing and vibration of the spectrograph.
    # Note that, this process can also be run after estimating the linecenter.
    if corInstShift:
        if refSpec is None:
            raise ValueError('refSpec is not given.')
        wvoffset = get_InstShift(data0, refSpec, dw=dwv)
    else:
        wvoffset = 0

    nw = ss.sum()
    wv = wv[ss].copy()
    if ndim == 3:
        data = data0[:,:,ss].copy()
    elif ndim == 2:
        data = data0[:,ss].copy()
    elif ndim == 1:
        data = data0[ss].copy() * np.ones((5,nw))

    ndim = data.ndim
    na = int(data.size/nw)
    data = data.reshape((na,nw))
    if iwc is None:
        s = data.argmin(axis=-1)
    else:
        s = np.abs(wv-iwc).argmin()*np.ones(na, dtype=int)
        # ds = int(abs(hw/dwv))
        # data[:,s-ds:s+ds]
        

    posi0 = np.arange(na)
    smin = [0,wv[0]]
    smax = [na-1,wv[-1]]
    order = [na,len(wv)]

    if method.lower() == 'linear':
        interp = LinearSpline(smin, smax, order, data)
    elif method.lower() == 'cubic':
        interp = CubicSpline(smin, smax, order, data)

    

    wc = np.zeros(na)
    hwc = np.zeros(na)
    intc = np.zeros(na)
    dwc = np.zeros(na)
    ref = 1
    rep = 0
    s0 = s.copy()
    more = data[posi0,s0] > 100

    if cm == 'ori':
        more = data[posi0,s0] > 100
        wl = np.array((posi0,wv[s]-hw)).T; wr = np.array((posi0,wv[s]+hw)).T
        intc = 0.5*(interp(wl)+interp(wr))
        while ref > 0.00001 and rep < 6 and more.sum() > 0:
            sp1 = data-intc[:,None]
            comp = sp1[:,0:nw-1]*sp1[:,1:nw]

            s = comp[more] <=0.
            nsol = s.sum(axis=1)
            j = nsol//2
            whl = nsol.cumsum()-nsol+j-1
            whr = nsol.cumsum()-nsol+j
            whp, whs = np.where(s)
            l = whs[whl]
            r = whs[whr]
            posi = posi0[more]
            wl0 = wv[l]-dwv/(sp1[posi,l+1]-sp1[posi,l])*sp1[posi,l]
            wr0 = wv[r]-dwv/(sp1[posi,r+1]-sp1[posi,r])*sp1[posi,r]
            wc[more] = 0.5*(wl0+wr0)
            hwc[more] = 0.5*np.abs(wr0-wl0)

            wl = np.array((posi,wc[more]-hw)).T; wr=np.array((posi,wc[more]+hw)).T
            intc[more] = 0.5*(interp(wl)+interp(wr))
            ref0 = np.abs(hwc-hw)
            ref = ref0.max()
            more = (ref0>0.00001)*(data[posi0,s0]>100)
            rep += 1


    elif cm == 'chae':
        wc = wv[s]
        wc0 = np.copy(wc)
        more = data[posi0,s0] > 0.
        residual = np.zeros(na)
        dresid = np.zeros(na)
        alpha = 1
        sigmaratio = (0.1*data.mean())/hw
        while  (ref > 1e-4 or rep < 5)  and rep <  30 and more.sum() > 0:
            posi = posi0[more]
            wls = np.array((posi, wc[more]-hw)).T
            wrs = np.array((posi, wc[more]+hw)).T

            delta = (interp(wrs)-interp(wls)) 
            delta1d = (interp(wrs+dwv)-interp(wrs-dwv) -interp(wls+dwv)+interp(wls-dwv))/(2*dwv)
            
    #        delta2d = (interp(wrs+eps)-2*interp(wrs)+interp(wrs-eps) \
    #                   -interp(wls+eps)+ 2*interp(wls)-interp(wls-eps))/eps**2
            residual[more] = delta * delta1d   \
                    + (wc[more] - wc0[more])*alpha*sigmaratio**2
            dresid[more] = delta1d**2 + alpha*sigmaratio**2
            dwc[more] = -0.4*residual[more]/dresid[more]
                
            
            wc[more] += dwc[more]
            ref0 = abs(dwc)
            ref = ref0.max()
            intc[more] = 0.5*(interp(wls)+interp(wrs))
            #if rep ==0:
            #    print('intc[more].shape=', intc[more].shape)
            more = (ref0 > 1e-4) 
            rep += 1
        # print(f'rep={rep:3.0f}, more.sum={more.sum():3.0f}')
            
            if rep > 2:
                alpha *= 0.1**(1./2.)
                
            alpha = np.maximum(alpha, 1.e-2)

    elif cm == 'nop': # Newton optimzation
        more = data[posi0,s0] > 10
        wc = wv[s0].copy()
        while ref >1e-4 and rep < 5 and more.sum() > 0:
            posi = posi0[more]
            wl = np.array((posi, wc[more]-hw)).T
            wr = np.array((posi ,wc[more]+hw)).T
            Fl = interp(wl)
            Fr = interp(wr) 
            Fr_p = interp(wr+dwv)
            Fr_n = interp(wr-dwv)
            Fl_p = interp(wl+dwv)
            Fl_n = interp(wl-dwv)
            F = Fr-Fl
            dF = (Fr_p-Fr_n - (Fl_p-Fl_n))/(2*dwv)
            dF2 = (Fr_p-2*Fr+Fr_n - (Fl_p-2*Fl+Fl_n))/dwv**2
            # f = F**2
            df = 2*F*dF
            df2 = 2*dF**2+2*F*dF2
            dwc[more] = -df/df2
            wc[more] += dwc[more]
            ref0 = abs(dwc)
            ref = ref0.max()
            intc[more] = 0.5*(Fr+Fl)
            more = ref0 > 1e-4
            rep += 1

    elif cm == 'nrf': # Newton root finding
        # sdata = gaussian_filter1d(data, 1, axis=1)
        # gdata = np.gradient(sdata, axis=1)
        # gdata = gaussian_filter1d(gdata, 1, axis=1)

        # if method.lower() == 'linear':
        #     ginterp = LinearSpline(smin, smax, order, gdata)
        # elif method.lower() == 'cubic':
        #     ginterp = CubicSpline(smin, smax, order, gdata)

        more = data[posi0,s0] > 10
        wc = wv[s0].copy()
        while ref >1e-4 and rep < 30 and more.sum() > 0:
            posi = posi0[more]
            wl = np.array((posi, wc[more]-hw)).T
            wr = np.array((posi ,wc[more]+hw)).T
            Fl = interp(wl)
            Fr = interp(wr) 
            Fr_p = interp(wr+dwv)
            Fr_n = interp(wr-dwv)
            Fl_p = interp(wl+dwv)
            Fl_n = interp(wl-dwv)
            F = Fr-Fl
            dFr = (Fr_p-Fr_n)/(2*dwv)
            dFl = (Fl_p-Fl_n)/(2*dwv)
            # dFl = ginterp(wl)
            # dFr = ginterp(wr)
            dF = dFr-dFl
            f = F**2
            df = 2*F*dF
            # sign1 = np.sign(F)
            # sign2 = np.sign(dFr)
            # sign3 = np.sign(dFl)
            # sign = -np.sign(sign1+sign2+sign3)
            # sign = -sign1*sign2
            # dwc[more] = sign*0.5*np.abs(f/(df+np.sign(df)))
            dwc[more] = -0.5*f/(df+np.sign(df))
            # dwc[more] = -f/df
            
            wc[more] += dwc[more]
            ref0 = abs(dwc)
            ref = ref0.max()
            intc[more] = 0.5*(Fr+Fl)
            more = ref0 > 1e-4
            rep += 1
    print(rep)
    if ndim == 1:
        wc = wc[3] - wvoffset
        intc = intc[3]
    else:
        wc = wc.reshape(reshape) - wvoffset
        intc = intc.reshape(reshape)
    return wc, intc
        
            
def _lmf(wc, par):
    interp = par['interp']
    hw = par['hw']
    posi = par['posi']
    wl = np.array((posi,wc[0]-hw)).T
    wr = np.array((posi,wc[0]+hw)).T
    
    Fl = interp(wl)
    Fr = interp(wr)
    F = np.abs(Fr-Fl)
    print(f"F:{F:.2f}, wc={wc[0]:.4f}")
    return F.max()

def LOS_velocity(wv,data,hw=0.01,band=False):
    """
    Calculte the Line-of-Sight velocity of given data.

    Parameters
    ----------
    wv : ~numpy.ndarray
        A Calibrated wavelength.
    data : ~numpy.ndarray
        n (n>=2) dimensional spectral profile data,
        the last dimension component must be the spectral component,
        and the size is equal to the size of wv.
    hw : float
        A half width of the horizontal line segment.
    band : str
        A string of the wavelength band.
        It must be the 4 characters in Angstrom unit. ex) '6562', '8542'

    Returns
    -------
    losv : ~numpy.ndarray
        n-1 (n>=2) dimensional Line-of_sight velocity value, where n is the
        dimension of the given data.

    Example
    -------
    >>> from fisspy.doppler import LOS_velocity
    >>> mask = np.abs(wv) < 1
    >>> losv = LOS_velocity(wv[mask],data[:,:,mask],hw=0.03,band='6562')
    """
    if not band :
        raise ValueError("Please insert the parameter band (str)")

    wc, intc =  lambdameter(wv,data,hw=hw,wvinput=True)

    if band == '6562' :
        return wc*c.to('km/s').value/6562.817
    elif band == '8542' :
        return wc*c.to('km/s').value/8542.09
    elif band == '5890' :
        return wc*c.to('km/s').value/5890.9399
    elif band == '5434' :
        return wc*c.to('km/s').value/5434.3398
    else:
        raise ValueError("Value of band must be one among"
                         "'6562', '8542', '5890', '5434'")
