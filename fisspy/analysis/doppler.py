from __future__ import absolute_import, division
import numpy as np
from interpolation.splines import LinearSpline, CubicSpline

__author__ = "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"

__all__ = ['lambdameter']


def lambdameter(wv, data0, hw=0.05, iwc=None, wvRange=None, cubic=False, rfm='hm', alpha=None, reguess=True, pr=False, corInstShift=False, refSpec=None):
    """
    Determine the Lambdameter chord center for a given half width or intensity.

    Parameters
    ----------
    wv: `~numpy.ndarray`
        A Calibrated wavelength.
    data: `~numpy.ndarray`
        n (n=2 or n=3) dimensional spectral profile data.
        The last dimension should be the spectral component,
        and the size is equal to that of wv.
    hw: `float`, optional
        A half width of the horizontal line segment.
    wvRange: `list`, optional
        Wavelength range to apply the lambdameter.
    iwc: `float`, optional
        Inital guess of the center of the lambdameter.
        Default is the minimum of the given at each line profile.
    cubic: `bool`, optional
        If True, using cubic interpolation to determine the function of the profile.
        If False, using linear interpolation.
        Default is False.
    rfm: `str`, optional
        Root-finding method.
        Default is 'nrm'
            - ori: bisector method
            - cm: Chae's method
            - nrm: Newton-Raphson method
            - hm: Halley's method
    alpha: `bool`, optional
        Stepsize of the root-finding.
        We recommand you does not touch this value except for using chae's method.
    reguess: `bool`, optional
        reguess the initial guess and repeat the calculation for the failed position to derive the line center by changing the intial guess automatically.
        Default is True.
    pr: `bool`, optional
        Print the result.
        Default is False.
    corInstShift: `bool`, optional
        If True, correct the instrument shift.
        Default is False
        Note, this process can also be done outside this function.
    refSpec: `float`, optional
        Reference spectrum for the correction of the instrument shift.
        If corInstShift is True, this should be given.

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
        This function was dratsically modified by J. Chae in 2023.
        This function was optimized by J. Kang in Mar 2024.

    Example
    -------
    >>> from fisspy.analysis import doppler
    >>> wc, inten = doppler.labdameter(wv,data,0.2)
    """
    # initial setting
    irfm = rfm.lower()
    shape = data0.shape
    reshape = shape[:-1]
    ndim = data0.ndim
    nw = shape[-1]
    if ndim >=4:
        raise ValueError('The dimension of data0 must be 2 or 3.')
    if wv.shape[0] != nw:
        raise ValueError('The number of elements of wv and '
        'the number of elements of last axis for data are not equal.')
    if hw <= 0:
        raise ValueError('The half-width value must be greater than 0.')
    
    dwv = wv[1]-wv[0]

    if wvRange is None:
        wvRange = [wv[0], wv[-1]]
    ss = (wv >= min(wvRange)) * (wv <= max(wvRange))

    nw = ss.sum()
    wv = wv[ss].copy()
    if ndim == 3:
        data = data0[:,:,ss].copy()
    elif ndim == 2:
        data = data0[:,ss].copy()
    elif ndim == 1:
        data = data0[ss].copy() * np.ones((5,nw))

    # flattening
    na = int(data.size/nw)
    data = data.reshape((na,nw))
    
    # calculation
    wc, intc, more = _LMminimization(wv, data, hw=hw, iwc=iwc, alpha=alpha, cubic=cubic, rfm=irfm)
    mm = more.sum()
    rep = 0

    if reguess and irfm != 'ori':
        # a = alpha
        # if a is None:
        #     a = 1
        if ndim >=3:
            for rep in range(3):
                if mm == 0:
                    break
                mm = 0
                wh = np.arange(na)
                wh = wh[more]
                for w in wh:
                    d = data[w]
                    d = d.copy() * np.ones((5,nw))
                    try:
                        iwc = np.median([wc[w-1:w+2], wc[w-1-shape[1]:w+2-shape[1]],
                                        wc[w-1+shape[1]:w+2+shape[1]]])
                    except:
                        iwc = np.median(wc)
                    res = _LMminimization(wv, d, hw=hw, iwc=iwc, alpha=alpha, cubic=cubic, rfm=irfm)
                    wc[w] = res[0][3]
                    intc[w] = res[1][3]
                    mm += res[2][3]
            if pr:
                print(f"#1st reguess: {rep}")
                print(f"#fail: {mm}")
            for rep2 in range(3):
                mwc = np.median(wc)
                v = (wc-mwc)/mwc*3e5
                msk = np.abs(v) > 40
                if msk.sum() == 0:
                    break
                wh = np.arange(na)
                wh = wh[msk]
                for w in wh:
                    d = data[w]
                    d = d.copy() * np.ones((5,nw))
                    try:
                        iwc = np.median([wc[w-1:w+2], wc[w-1-shape[1]:w+2-shape[1]],
                                        wc[w-1+shape[1]:w+2+shape[1]]])
                    except:
                        iwc = np.median(wc)
                    res = _LMminimization(wv, d, hw=hw, iwc=iwc, alpha=alpha, cubic=cubic, rfm=irfm)
                    wc[w] = res[0][3]
                    intc[w] = res[1][3]
            if pr:
                msk = np.abs(wc-np.median(wc)) > 1.5
                print(f"#2nd reguess: {rep2}")
                print(f"#abnormal pixels: {msk.sum()}")
    else:
        if pr:
            print(f"#fail: {mm}")


    # correct instrumental shift by the seeing and vibration of the spectrograph.
    # Note that, this process can also be run after estimating the linecenter.
    if corInstShift:
        from ..correction.correction import get_InstShift
        if refSpec is None:
            raise ValueError('refSpec is not given.')
        wvoffset = get_InstShift(data0, refSpec, dw=dwv)
    else:
        wvoffset = 0

    if ndim == 1:
        wc = wc[3] - wvoffset
        intc = intc[3]
    else:
        wc = wc.reshape(reshape) - wvoffset
        intc = intc.reshape(reshape)

    # if pr:
    #     return wc, intc, more.reshape(reshape)
    # if pr:
    #     print(f"rep: {rep}, #fail:{more.sum()}")
    #     if cm =='nm':
    #         return wc, intc, more.reshape(reshape), wc2.reshape([shape[0], shape[1], 30])
    #     return wc, intc, more.reshape(reshape)
    return wc, intc
        
def  _LMminimization(wv, data, hw=0.05, iwc=None, alpha=None, cubic=False, rfm='nrm'):
    """
    ori: bisector method
    cm: Chae's method
    nrm: Newton-Raphson method
    hm: Halley's method
    """
    shape = data.shape
    na, nw = shape
    dwv = wv[1]-wv[0]
    posi0 = np.arange(na)

    # create interpolation function
    smin = [0,wv[0]]
    smax = [na-1,wv[-1]]
    order = [na,len(wv)]
    if cubic:
        interp = CubicSpline(smin, smax, order, data)
    else:
        interp = LinearSpline(smin, smax, order, data)

    if iwc is None:
        s = data.argmin(axis=-1)
    else:
        s = np.abs(wv-iwc).argmin()*np.ones(na, dtype=int)

    wc = np.zeros(na)
    hwc = np.zeros(na)
    intc = np.zeros(na)
    dwc = np.zeros(na)
    ref = 1
    rep = 0
    s0 = s.copy()
    more = data[posi0,s0] > 10


    if rfm == 'ori':
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

    elif rfm == 'cm':
        wc = wv[s]
        wc0 = np.copy(wc)
        more = data[posi0,s0] > 0.
        residual = np.zeros(na)
        dresid = np.zeros(na)
        if alpha is None:
            Alpha = 1
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
                    + (wc[more] - wc0[more])*Alpha*sigmaratio**2
            dresid[more] = delta1d**2 + Alpha*sigmaratio**2
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
                Alpha *= 0.1**(1./2.)
                
            Alpha = np.maximum(Alpha, 1.e-2)

    elif rfm == 'nrm': # Newton-Raphson Method
        if alpha is None:
            Alpha = 1
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
            dwc[more] = -Alpha*f/(df+np.sign(df)*1e-3)
            # dwc[more] = -f/df
            
            wc[more] += dwc[more]
            ref0 = abs(dwc)
            ref = ref0.max()
            intc[more] = 0.5*(Fr+Fl)
            more = ref0 > 1e-4
            rep += 1
            
    elif rfm == 'hm': # Halley's method (modified NRM)
        if alpha is None:
            Alpha = 1
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
            dF = dFr-dFl
            dF2 = (Fr_p-2*Fr+Fr_n - (Fl_p-2*Fl+Fl_n))/dwv**2
            f = F**2
            df = 2*F*dF
            df2 = 2*(dF**2+F*dF2)
            denom = (df**2-0.5*f*df2)
            dwc[more] = -Alpha*f*df/(denom+np.sign(denom)*1e-3)
            
            wc[more] += dwc[more]
            ref0 = abs(dwc)
            ref = ref0.max()
            intc[more] = 0.5*(Fr+Fl)
            more = ref0 > 1e-4
            rep += 1
    else:
        raise ValueError("rfm should be one among 'ori', 'cm', 'nrm', 'hm'.")
    
    wc[more] = 99
    intc[more]= 0
    wc = np.nan_to_num(wc, nan=99, posinf=99, neginf=99)
    intc = np.nan_to_num(intc, nan=0, posinf=0, neginf=0)
    return wc, intc, more
