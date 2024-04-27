import numpy as np
from astropy.io import fits
from interpolation.splines import LinearSpline, CubicSpline
from fisspy.align import shiftImage, alignOffset, rotImage
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from os.path import join, isdir, isfile, dirname, basename, abspath
from os import getcwd, makedirs
from glob import glob
from astropy.time import Time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from ..analysis.wavelet import Wavelet
from scipy.fftpack import fft, ifft
from urllib.request import urlretrieve

def fname2isot(f):
    """
    Translate the file name in to isot.

    Parameters
    ----------
    f: `str`
        filename.

    Returns
    -------
    isot: `str`
        datetime in the form of isot.
    """
    rf = basename(f).replace('_BiasDark', '')
    rf = basename(rf).replace('_FLAT', '')
    rf = basename(rf).replace('_xFringe', '')
    rf = basename(rf).replace('_yFringe', '')
    sp = rf.split('_')
    YY = sp[1][:4]
    MM = sp[1][4:6]
    DD = sp[1][6:8]

    hh = sp[2][:2]
    mm = sp[2][2:4]
    ss = sp[2][4:]

    return f"{YY}-{MM}-{DD}T{hh}:{mm}:{ss}"

def multiGaussian(x, *pars):
    """
    Get multi-Gaussian function
    """
    ng = len(pars)//3
    y = np.zeros((len(x)))
    for i in range(ng):
        A = pars[i*3]
        m = pars[i*3+1]
        sig = pars[i*3+2]
        y += A*np.exp(-((x-m)/sig)**2)
    return y

def Gaussian(x, *par):
    """
    Get Gaussian function
    """
    A = par[0]
    m = par[1]
    sig = par[2]
    y = A*np.exp(-((x-m)/sig)**2)
    return y

def getMask(data, power=4, fsig=1.2, **kwags):
    """
    Get spectral mask

    Parameters
    ----------
    data: `~numpy.ndarray`
        2D spectrogram
    power: `float` (optional)
        depth factor of the mask.
        Default is 4.
    fsig: `float` (optional)
        width factor of the mask.
        Default is 1.2

    Returns
    -------
    mask: `~numpy.ndarray`
        mask.
    """
    wl = kwags.pop('window_length', 10)
    po = kwags.pop('polyorder', 2)
    dv = kwags.pop('deriv', 0)
    delta = kwags.pop('delta', 1.0)
    mode = kwags.pop('mode', 'interp')
    cval = kwags.pop('cval', 0.0)

    sh = data.shape
    tmp0 = data[:,10:-10].mean(1)[:,None,:]*np.ones(sh)
    tmp = tmp0[...,5:-5]
    der2 = np.gradient(np.gradient(tmp, axis=2), axis=2)
    der2 -= der2[:,10:-10,10:-10].mean((1,2))[:,None, None]
    std = der2[:,10:-10,10:-10].std((1,2))[:,None,None]
    msk = np.ones(sh, dtype='float')
    msk[...,5:-5] = np.exp(-0.5*np.abs((der2/std))**2)
    msk[...,5:-5] = savgol_filter(msk[...,5:-5], wl, po,
                      deriv= dv, delta= delta, cval= cval,
                      mode= mode, axis=2)
    msk[msk > 1] = 1
    msk[...,:5] = msk[...,6][...,None]
    msk[...,-5:] = msk[...,-6][...,None]
    for i in range(sh[0]):
        ttmp = tmp0[i,100]
        mm = ttmp[10:-10].argmin()+10
        lb = mm-10
        if lb < 0:
            lb = 0
        rb = mm+10
        if rb >= sh[2]:
            rb = sh[2]-1
        x = np.arange(lb,rb)
        c = (ttmp[10:50].mean()+ttmp[-50:-10].mean())/2
        cp, cr = curve_fit(Gaussian, x, ttmp[lb:rb]-c, p0=[ttmp[mm], mm, 5])
        msk[i,:,int(cp[1]-np.abs(cp[2])*fsig):int(cp[1]+np.abs(cp[2])*fsig)] = 0
    return msk**power

def data_mask_and_fill(data, msk_range, axis=1, kind='nearest'):
    shape = data.shape
    minMsk, maxMsk = msk_range
    nmsk = len(minMsk)
    if nmsk != len(maxMsk):
        raise(ValueError("The number of minimum mask range is different from the maximum mask range"))
    x = np.arange(shape[axis])
    xt = x.copy()
    tdata = data.copy()

    lmsk = 0
    for i in range(nmsk):
        xt = np.delete(xt, np.arange(lmsk+minMsk[i], lmsk+maxMsk[i]))
        tdata = np.delete(tdata, np.arange(lmsk+minMsk[i], lmsk+maxMsk[i]), axis=axis)
        lmsk -= maxMsk[i] - minMsk[i]
        
    interp = interp1d(xt, tdata, axis=axis, kind=kind)
    mdata = interp(x)
    return mdata

def cal_fringeGauss(wvlet, filterRange=[0,-1]):
    """
    axis: `int`, optional
        0, wavelet along the slit direction
    """
    shape = wvlet.wavelet.shape
    nwl = shape[1]
    for i in range(2):
        if filterRange[i] < 0:
            filterRange[i] = nwl + filterRange[i]

    x = np.arange(nwl)

    pars = [None]*3
    
    freq = np.arctan2(wvlet.wavelet.imag, wvlet.wavelet.real)
    coeff = np.zeros((3, shape[0], shape[2]))
    Awvlet = np.abs(wvlet.wavelet)
    fringe_wvlet = np.zeros(shape, dtype=complex)

    for ii in range(shape[0]):
        # if ii % 10 == 0:
        #     print(f"calculate {ii}-th row")
        wh = None
        for jj in range(shape[2]):
            wv = Awvlet[ii,:,jj]
            pars[0] = wv[filterRange[0]:filterRange[1]].max()
            if wh is None:
                wh = wv[filterRange[0]:filterRange[1]].argmax() + filterRange[0]
            else:
                wh = wv[wh-3:wh+3].argmax() + wh-3
            pars[1] = wh
            pars[2] = 2
            try:
                cp, cov = curve_fit(Gaussian, x[wh-5:wh+5], wv[wh-5:wh+5], p0=pars)
                coeff[:,ii,jj] = cp
            except:
                print(f"catch err at {ii},:,{jj}")
                return None

    fringe_pwr = Gaussian(x[None,:,None], *coeff[:,:,None,:])
    fringe_wvlet = fringe_pwr*(np.cos(freq)+1j*np.sin(freq))
    fringe = wvlet.iwavelet(fringe_wvlet, wvlet.scale)

    return fringe

def cal_fringeSimple(wvlet, filterRange):
    wavelet = wvlet.wavelet.copy()
    shape = wavelet.shape
    
    nwl = shape[1]

    wavelet[:,:filterRange[0]] = 0
    wavelet[:,filterRange[1]:] = 0

    return wvlet.iwavelet(wavelet, wvlet.scale)
    
def get_tilt_old(img, tilt=None, show=False):
    """
    Get a tilt angle of the spectrum camera in the unit of degree.

    Parameters
    ----------
    img : `~numpy.array`
        A two-dimensional `~numpy.array` of the form ``(y, x)``.
    show : `bool`, optional
        If `False` (default) just calculate the tilt angle.
        If `True` calculate and draw the original image and the rotation corrected image.

    Returns
    -------
    Tilt : `float`
        A tilt angle of the spectrum camera in the unit of degree.

    Examples
    --------
    >>> from astropy.io import fits
    >>> data = fits.getdata("FISS_20140603_164020_A_Flat.fts")[3]
    >>> tilt = get_tilt(data, show=True)
    """

    nw = img.shape[-1]
    wp = 40

    dy_img = np.gradient(img, axis=0)
    whd = np.abs(dy_img[20:-20,wp:wp+20].mean(1)).argmax() + 20
    i1 = dy_img[whd-16:whd+16, wp:wp+16]
    i2 = dy_img[whd-16:whd+16, -(wp+16):-wp]
    
    tmp = 10
    shy = 0
    iteration = 0
    while abs(tmp) > 1e-1:
        sh = alignOffset(i2, i1)
        sh[1,0] = 0
        tmp = sh[0,0]
        shy += tmp
        # i2 = shift(i2, -sh, missing=-1, cubic=True)
        i2 = shiftImage(i2, -sh, missing=None, cubic=True)
        iteration += 1
        if iteration == 10:
            print('break')
            break

    
    if tilt is None:
        Tilt = np.rad2deg(np.arctan2(shy, nw - wp*2))
    else:
        Tilt = tilt


    if show:
        rimg = rotImage(img, np.deg2rad(-Tilt), cubic=True, missing=None)
        
        fig, ax = plt.subplots(2,1, figsize=[6, 6], sharey=True, sharex=True)
        iimg = img - np.median(img, axis=0)
        m = iimg[whd-16:whd+16].mean()
        std = iimg[whd-16:whd+16].std()
        imo = ax[0].imshow(iimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        imo.set_clim(m-std*2,m+std*2)
        
        irimg = rimg - np.median(rimg, axis=0)
        imr = ax[1].imshow(irimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        clim = imo.get_clim()
        imr.set_clim(clim)
        # imr.set_clim(m-std,m+std)

        # ax[0].set_xlabel('Wavelength (pix)')
        ax[1].set_xlabel('Wavelength (pix)')
        ax[1].set_ylabel('Slit (pix)')
        ax[0].set_ylabel('Slit (pix)')
        ax[0].set_title('tilted image')
        ax[1].set_title('corrected image')

        ax[0].set_ylim(whd-10,whd+10)

        ax[0].set_aspect(adjustable='box', aspect='auto')
        ax[1].set_aspect(adjustable='box', aspect='auto')

        fig.tight_layout()
        fig.show()
        try:
            fig.canvas.manager.window.move(0,0)
        except:
            pass

    return Tilt

def get_tilt(img, tilt=None, show=False):
    """
    Get a tilt angle of the spectrum camera in the unit of degree.

    Parameters
    ----------
    img : `~numpy.array`
        A two-dimensional `~numpy.array` of the form ``(y, x)``.
    show : `bool`, optional
        If `False` (default) just calculate the tilt angle.
        If `True` calculate and draw the original image and the rotation corrected image.

    Returns
    -------
    Tilt : `float`
        A tilt angle of the spectrum camera in the unit of degree.

    Examples
    --------
    >>> from astropy.io import fits
    >>> data = fits.getdata("FISS_20140603_164020_A_Flat.fts")[3]
    >>> tilt = get_tilt(data, show=True)
    """

    nw = img.shape[-1]
    wp = 40

    dy_img = np.gradient(np.gradient(img, axis=0), axis=0)
    k = np.abs(dy_img[20:-20,wp:wp+20].mean(1))
    pks = find_peaks(k, k.std()*2)[0]+20
    npks = len(pks)

    shy = 0
    for whd in pks:
        i1 = dy_img[whd-16:whd+16, wp:wp+16]
        i2 = dy_img[whd-16:whd+16, -(wp+16):-wp]
        sh = alignOffset(i2, i1)
        shy += sh[0,0]
    shy /= npks
    
    if tilt is None:
        Tilt = np.rad2deg(np.arctan2(shy, nw - wp*2))
    else:
        Tilt = tilt


    if show:
        rimg = rotImage(img, np.deg2rad(-Tilt), cubic=True, missing=None)
        
        fig, ax = plt.subplots(2,1, figsize=[6, 6], sharey=True, sharex=True)
        iimg = img - np.median(img, axis=0)
        m = iimg[whd-16:whd+16].mean()
        std = iimg[whd-16:whd+16].std()
        imo = ax[0].imshow(iimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        imo.set_clim(m-std*2,m+std*2)
        
        irimg = rimg - np.median(rimg, axis=0)
        imr = ax[1].imshow(irimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        clim = imo.get_clim()
        imr.set_clim(clim)
        # imr.set_clim(m-std,m+std)

        # ax[0].set_xlabel('Wavelength (pix)')
        ax[1].set_xlabel('Wavelength (pix)')
        ax[1].set_ylabel('Slit (pix)')
        ax[0].set_ylabel('Slit (pix)')
        ax[0].set_title('tilted image')
        ax[1].set_title('corrected image')

        ax[0].set_ylim(whd-10,whd+10)

        ax[0].set_aspect(adjustable='box', aspect='auto')
        ax[1].set_aspect(adjustable='box', aspect='auto')

        fig.tight_layout()
        fig.show()
        try:
            fig.canvas.manager.window.move(0,0)
        except:
            pass

    return Tilt

def piecewise_quadratic_fit(x, y, npoint=5):
    """
    Smoothing the data by applying the piecewise quadratic fit.

    Paramteres
    ----------
    x: `~numpy.array`
        An one-dimensional `~numpy.array`.
    y: `~numpy.array`
        An one-dimensional `~numpy.array` to be filtered.
    npoint: `int`
        The length of the filter window.

    Returns
    -------
    yf: `~numpy.array`
        The filtered data.
    """
    nx = len(x)
    m = npoint if npoint < nx else nx
    yf = y.copy()
    for i in range(nx):
        sid = int(i - m//2)
        sid = sid if sid > 0 else 0
        fid = sid+m
        fid = fid if fid < nx else nx
        sid = fid-m
        coeff = np.polyfit(x[sid:fid], y[sid:fid], 2)
        yf[i] = np.polyval(coeff, x[i])
    return yf

def get_curve_par(cData, show=False):
    nDim = cData.ndim
    ones = np.ones((4,16))
    if nDim == 3:
        nf, ny, nw = cData.shape
        d2Data = np.gradient(np.gradient(cData[:,5:-5,5:-5], axis=2), axis=2)

    elif nDim == 2:
        ny, nw = cData.shape
        dw = np.zeros(ny)
        d2Data = np.gradient(np.gradient(cData[:,5:-5], axis=1), axis=1)
        nf = 1
        d2Data = d2Data[None,:,:]
        
    dw = np.zeros((nf,ny))
    for f in range(nf):
        k = d2Data[f].mean(0)
        pk = find_peaks(k, k.std()*2)[0]
        pk = pk[(pk >= 8) * (pk + 8 <=nw-15)]
        npk = len(pk)
        dwpk = np.zeros((npk, ny))
        for pp, wh in enumerate(pk):
            for direction in range(-1,2,2):
                prof0 = d2Data[f,ny//2,wh-8:wh+8]*ones
                for i, prof in enumerate(d2Data[f, ny//2 + direction::direction]):
                    prof = prof[wh-8:wh+8]*ones
                    dwpk[pp,direction*(i+1) + ny//2] = alignOffset(prof, prof0)[1]
                    dwpk[pp,direction*(i+1) + ny//2] += dwpk[pp,direction*(i) + ny//2]
                    prof0 = prof
        dw[f] = np.median(dwpk,axis=0)

    dw = dw.mean(0)

    y = np.arange(ny)
    p = np.polyfit(y[10:-10], dw[10:-10], 2)

    if show:
        fig, ax = plt.subplots(figsize=[6,4])
        wf = np.polyval(p, y)
        ax.scatter(y, dw, marker='+')
        p1 = f"$+{p[1]:.2e}x$" if np.sign(p[1]) == 1 else f"${p[1]:.2e}x$"
        p2 = f"$+{p[2]:.2e}" if np.sign(p[2]) == 1 else f"${p[2]:.2e}"
        eq = f"$y = {p[0]:.2e}x^2${p1}{p2}"
        eq = eq.replace('e','^{')
        eq = eq.replace('x','}x')
        eq = eq + '}$'
        ax.plot(y, wf, color='r', label=eq)
        ax.set_xlabel('Slit (pix)')
        ax.set_ylabel('dw (pix)')
        ax.set_title('Curvature')
        ax.legend()
        fig.tight_layout()
        fig.show()
        try:
            fig.canvas.manager.window.move(1300,0)
        except:
            pass

    return p, dw

def tilt_correction(img, tilt, cubic=True):
    """
    Return the tilt corrected image.

    Paramters
    ---------
    img: `~numpy.array`
        N-dimensional `~numpy.array` the last dimension should be the wavelength and the -2 dimension should be the slit dimension.
        This image should be dark bias corrected, flat-fielded.
    tilt: `float`
        Tilt angle of the image in degree unit.
    cubic: `bool`, optional
        If `True` (default), apply the cubic interpolation to rotate the image.
        If `False`, apply the linear interpolation.

    Returns
    -------
    ti: `~numpy.array`
        N-dimensional `~numpy.array` of tilt corrected image. 
    """
    ti = rotImage(img, np.deg2rad(-tilt), cubic=cubic, missing=None)

    return ti

def curvature_correction(img, coeff, show=False):
    """
    Return the curvature corrected image.

    Parameters
    ----------
    img: `~numpy.array`
        N-dimensional `~numpy.array` the last dimension should be the wavelength and the -2 dimension should be the slit dimension.
        This image should be dark bias corrected, flat-fielded, and tilt corrected.
    coeff: `list`
        Results of the `get_curve_par` function
    show: `bool`, optional
        If `True`, draw raw and curvature corrected image. Default is `False`

    Returns
    -------
    ccImg: `~numpy.array`
        N-dimensional `~numpy.array` of curvature corrected image.
    """
    shape = img.shape
    ndim = img.ndim
    size = img.size
    y = np.arange(shape[-2])
    dw = np.polyval(coeff, y)

    dw -= (dw.min() + dw.max())/2
    order = shape
    smin = np.zeros(ndim)
    smax = np.array(shape)-1
    interp = CubicSpline(smin, smax, order, img)

    ones = np.ones(shape)
    w = (np.arange(shape[-1]))[tuple([None]*(ndim-1) + [Ellipsis])]*ones + dw[tuple([None]*(ndim-2) + [Ellipsis] + [None])]
    y = np.arange(shape[-2])[tuple([None]*(ndim-2) + [Ellipsis] + [None])]*ones
    inp = np.zeros((ndim,size))
    idx = [None]*(ndim-2)
    for i, sh in enumerate(shape[:-2]):
        tmp = np.arange(sh)[tuple([None]*i + [Ellipsis] + [None]*(ndim-1-i))]*ones
        inp[i] = tmp.reshape(size)
        idx[i] = sh//2
    inp[-1] = w.reshape(size)
    inp[-2] = y.reshape(size)

    ccImg = interp(inp.T).reshape(shape)

    if show:
        oimg = img[tuple(idx)].squeeze()
        prof = oimg[20]
        dp2 = np.gradient(np.gradient(prof))
        wh = dp2.argmax()
        cimg = ccImg[tuple(idx)].squeeze()

        return ccImg, oimg, cimg, wh
    else:
        return ccImg
    
class calFlat:
    def __init__(self, fflat, ffoc=None, tilt=None, autorun=True, shiftcor=True, save=True, show=False, maxiter=10, msk=None):
        """
        Make the master flat and slit pattern data.

        Parameters
        ----------
        fflat: `str`
            Raw flat file.
        ffoc: `str`, optional.
            If ffoc is given, the tilt angle is cacluate by using the focus pinhole image of ffoc file.
            If `None` default, the tilt angle is calculate by using the flat file itself.
        tilt: `float`, optional
            Tilt angle of the image in degree unit.
            If `None` (default), calculate the tilt angle by using the flat or focus file.
        autorun: `bool`, optional
            If `True` (default), run make_slit, gain calib and save fits (optional) directly.
        save: `bool`, optional
            If `True` (default), save the master flat and slit pattern when autorun is `True`.
        show: `bool`, optional
            If `True`, draw raw and flat-fielded image. Default is `False`
        maxiter: `int`, optional
            Maximum number of iteration to repeat the calculation for creation master flat.
        msk: `~numpy.array`, optional
            Spectrum Mask
        
        Returns
        -------
        calFlat instance.
        """
        opn = fits.open(fflat)[0]
        self.h = opn.header
        self.rawFlat = opn.data
        self.date = self.h['date']
        self.nf, self.ny, self.nw =  self.rawFlat.shape
        self.logRF = np.log10(self.rawFlat)
        self.mlogRF = self.logRF.mean(0)
        self.tilt = tilt
        self.ffoc = ffoc
        self.fflat = fflat
        self.shyA = None
        self.fsData = None
        self.logF2 = None
        
        fdir = dirname(fflat)
        if not fdir:
            fdir = getcwd()

        self.sdir = join(dirname(fdir), 'proc', 'cal')
        # get tilt angle in degree
        # if tilt is None:
        #     if ffoc is not None:
        #         foc = fits.getdata(ffoc)
        #         self.mfoc = foc.mean(0)
        #         self.tilt = get_tilt(self.mfoc, show=show)

        #     else:
        #         self.tilt = get_tilt(10**self.mlogRF, tilt=tilt, show=show)

        # print(f"Tilt: {self.tilt:.3f} degree")
        

        
        # if autorun:
        #     self.rlRF = tilt_correction(self.logRF, self.tilt, cubic=True)
        #     # get slit pattern
        #     self.Slit = self.make_slit_pattern(cubic=True, show=show, shiftcor=shiftcor)
        #     # remove the slit pattern
        #     if self.shiftcor:
        #         self.logF = np.zeros((self.nf, self.ny, self.nw))
        #         sh = np.zeros((2,1))
        #         for i in range(self.nf):
        #                 sh[0,0] = self.shyA[i]
        #                 sslit = shift(self.Slit, -sh, missing=-1, cubic=True)
        #                 self.logF[i] = self.rlRF[i] - np.log10(sslit)
        #     else:
        #         self.logF = self.rlRF - np.log10(self.Slit)

        #     # curvature correction
        #     self.coeff = coeff = get_curve_par(self.logF, show=show)
        #     self.logF = curvature_correction(self.logF, coeff, show=show)
        #     plt.pause(0.1)

        #     # fringe subtraction
        #     self.atlas_subtraction()
            

        #     self.Flat = self.gain_calib(maxiter=maxiter, msk=msk, show=show)

        #     # corrected flat pattern
        #     self.cFlat = 10**(self.logF - np.log10(self.Flat))
        #     if save:
        #         self.saveFits(self.sdir)
        
            
        #     if show:
        #         fig, ax = plt.subplots(2, figsize=[7,7], sharex=True, sharey=True)
        #         im0 = ax[0].imshow(10**self.logRF[3], plt.cm.gray, origin='lower', interpolation='bilinear')
        #         m = (10**self.logRF[3]).mean()
        #         std = (10**self.logRF[3]).std()
        #         im0.set_clim(m-std, m+std)
        #         ax[0].set_ylabel("Slit (pix)")
        #         ax[0].set_title("Raw Data")
        #         logf = np.log10(self.Flat)
        #         im = ax[1].imshow(self.cFlat[3], plt.cm.gray, origin='lower', interpolation='bilinear')
        #         # im.set_clim(self.cFlat[3][10:-10,10:-10].min(),self.cFlat[3][10:-10,10:-10].max())
        #         m = self.cFlat[3][10:-10,10:-10].mean()
        #         std = self.cFlat[3][10:-10,10:-10].std()
        #         self.im = im
        #         im.set_clim(m-std, m+std)
        #         ax[1].set_xlabel("Wavelength (pix)")
        #         ax[1].set_ylabel("Slit (pix)")
        #         ax[1].set_title("Flat correction")
        #         fig.tight_layout()
        #         fig.show()
        #         try:
        #             fig.canvas.manager.window.move(600,0)
        #         except:
        #             pass

    def make_slit_pattern(self, shiftcor=True, cubic=True, show=False):
        """
        Make the slit pattern image with the given tilt angle.

        Paramters
        ---------
        cubic: `bool`, optional
            If `True` (default), apply the cubic interpolation to rotate the image.
            If `False`, apply the linear interpolation.
        show: `bool, optional
            If `True`, Draw a slit pattern image. Default is `False`

        Returns
        -------
        Slit: `~numpy.array`
        """
        self.shiftcor = shiftcor
        
        if shiftcor:
            d2rlRF = np.gradient(np.gradient(self.rlRF, axis=1), axis=1)
            wh = int(np.median(np.abs(d2rlRF[self.nf//2,5:-5,5:-5]).argmax(0))) + 5
            ref = d2rlRF[self.nf//2,wh-16:wh+16,5:-5]
            si = np.zeros((self.nf, self.ny, self.nw))
            si[self.nf//2] = self.rlRF[self.nf//2]
            self.shyA = np.zeros(self.nf)
            for i in range(self.nf):
                if i == self.nf//2:
                    continue
                spec = d2rlRF[i,wh-16:wh+16,5:-5]
                sh = alignOffset(spec, ref)
                sh[1,0] = 0
                self.shyA[i] = -sh[0,0]
                si[i] = shiftImage(self.rlRF[i], -sh, missing=None, cubic=True)
        else:
            si = self.rlRF
        
        ri = np.median(si, axis=0)
        Slit = 10**np.median(ri[:,40:-40], axis=1)[:,None] * np.ones([self.ny,self.nw])
        Slit /= np.median(Slit)
        
        self.tsi = si
        self.tfig = None

        if show:
            fig, ax = plt.subplots(figsize=[6,3.5])
            im = ax.imshow(Slit, plt.cm.gray, origin='lower', interpolation='bilinear')
            ax.set_xlabel("Wavelength (pix)")
            ax.set_ylabel("Slit (pix)")
            ax.set_title(f"Slit Pattern")
            fig.tight_layout()
            fig.show()
            try:
                fig.canvas.manager.window.move(0,1080-350)
            except:
                pass
            
        return Slit


    def gain_calib(self, idata, maxiter=10, msk=None, show=False):
        """
        Make the master flat following the technique decribed in Chae et al. (2013).

        Parameters
        ----------
        maxiter: `int`, optional
            Maximum number of iteration to repeat the calculation for creation master flat.
        msk: `~numpy.array`, optional
            Spectrum Mask
        show: `bool`, optional
            If `True`, Draw a Flat image. Default is `False`

        Returns
        -------
        Flat: `~numpy.array`
            Master Flat
        """

        if self.logF2 is None:
            logF =  self.logF
        else:
            logF = self.logF2

        if msk is None:
            msk = getMask(10**logF, power=4)
        self.msk = msk
        
        # self.rmFlat2 = self.rmFlat + self.mlf.max(0) # y direction vignetting is removed (that is not intended problems)
        tt = idata
        self.C = (tt*msk).sum((1,2))/msk.sum((1,2))
        self.C -= self.C.mean()

        # Flat = np.median(self.logF, axis=0)
        # Flat -= np.median(Flat)
        Flat = tt.mean(0)
        Flat -= np.mean(Flat)
        f1d = np.gradient(Flat, axis=1)
        f2d = np.gradient(f1d, axis=1)
        mask = (np.abs(f2d) <= f2d.std()) * (np.abs(f1d) <= f1d.std())
        mask[:,100:-100] = False

        w = np.arange(self.nw)
        for i, m in enumerate(mask):
            coeff = np.polyfit(w[m], Flat[i,m], 2)
            Flat[i] = np.polyval(coeff, w)

        self.x = np.zeros(self.nf)
        self.xi = np.zeros(self.nf)
        xdum = np.arange(self.nw, dtype=int)
        hy = int(self.ny//2)
        one = np.ones((4, self.nw))

        for k in range(self.nf):
            self.xi[k] = xdum[tt[k, hy] == tt[k, hy, 5:-5].min()][0]

        for k in range(self.nf-1):
            img1 = (logF[k+1] - Flat)[hy-10:hy+10].mean(0)*one
            img2 = (logF[k] - Flat)[hy-10:hy+10].mean(0)*one
            sh = alignOffset(img1, img2)
            dx = int(np.round(sh[1]))
            if dx < 0:
                img1 = (logF[k+1] - Flat)[hy-10:hy+10, :dx].mean(0)*one[:,:dx]
                img2 = (logF[k] - Flat)[hy-10:hy+10, -dx:].mean(0)*one[:,-dx:]
                sh, cor = alignOffset(img1, img2, cor=True)
            else:
                img1 = (logF[k+1] - Flat)[hy-10:hy+10, dx:].mean(0)*one[:,dx:]
                img2 = (logF[k] - Flat)[hy-10:hy+10, :-dx].mean(0)*one[:,:-dx]
                sh, cor = alignOffset(img1, img2, cor=True)
            self.x[k+1] = self.x[k] + sh[1] + dx
            # print(f"k: {k+1}, x={self.x[k+1]}, cor={cor}")
        self.x -= np.median(self.x)


        self.dx = np.zeros([self.nf, self.ny])
        y = np.arange(self.ny)
        for k in range(self.nf):
            self.ref = np.gradient(np.gradient((logF[k]-Flat)[hy-10:hy+10].mean(0), axis=0), axis=0)*one
            for j in range(self.ny):
                img = np.gradient(np.gradient((logF[k] - Flat)[j], axis=0), axis=0)*one
                sh = alignOffset(img[:,5:-5], self.ref[:,5:-5])
                self.dx[k,j] = sh[1]
            self.dx[k] = piecewise_quadratic_fit(y, self.dx[k], 100)


        shape = [self.nf, self.ny, self.nw]
        ones = np.ones(shape)
        size = ones.size
        y = np.arange(self.ny)[None,:,None]*ones
        f = np.arange(self.nf)[:,None,None]*ones
        pos = np.arange(self.nw)[None,None,:] + self.x[:,None,None] + self.dx[:,:,None]
        weight = (pos >= 0) * (pos < self.nw)
        pos[pos < 0] = 0
        pos[pos > self.nw-1] = self.nw-1
        data = tt - Flat
        smin = [0, 0, 0]
        smax = [self.nf-1, self.ny-1, self.nw-1]
        order = [self.nf, self.ny, self.nw]
        interp = CubicSpline(smin, smax, order, data)
        inp = np.array((f.reshape(size), y.reshape(size), pos.reshape(size)))
        a = interp(inp.T).reshape(shape)
        a = a*weight
        b = weight.sum((0,1))
        b[b < 1] = 1
        self.Object = a.sum((0,1))/b
        
        for i in range(maxiter):
            pos = np.arange(self.nw)[None,None,:] - self.x[:,None,None] - self.dx[:,:,None]
            weight = (pos >= 0) * (pos < self.nw)
            pos[pos < 0] = 0
            pos[pos > self.nw-1] = self.nw-1
            weight = weight* msk
            interp = LinearSpline(smin, smax, order, self.Object*ones)
            inp = np.array((f.reshape(size), y.reshape(size), pos.reshape(size)))
            obj1 = interp(inp.T).reshape(shape)
            ob = (self.C[:,None,None] + obj1 + Flat - tt)*weight
            self.C -= ob.sum((1,2))/weight.sum((1,2))
            data = np.gradient(self.Object, axis=0)*ones
            interp = LinearSpline(smin, smax, order, data)
            oi = -interp(inp.T).reshape(shape)
            self.x -= (ob*oi).sum((1,2))/(weight*oi**2).sum((1,2))
            b = weight.sum(0)
            b[b < 1] = 1
            DelFlat = -ob.sum(0)/b
            Flat += DelFlat

            pos = np.arange(self.nw)[None,None,:] + self.x[:,None,None] + self.dx[:,:,None]
            weight = (pos >= 0) * (pos < self.nw)
            pos[pos < 0] = 0
            pos[pos > self.nw-1] = self.nw-1
            interp = LinearSpline(smin, smax, order, self.msk)
            inp = np.array((f.reshape(size), y.reshape(size), pos.reshape(size)))
            weight = weight*interp(inp.T).reshape(shape)
            data = tt - Flat
            interp = CubicSpline(smin, smax, order, data)
            a = (self.C[:, None, None] + self.Object[None,None,:] - interp(inp.T).reshape(shape))*weight

            b = weight.sum((0,1))
            b[b < 1] = 1
            DelObject = - a.sum((0,1))/b
            self.Object += DelObject

            err = np.abs(DelFlat).max()
            print(f"iteration={i}, err: {err:.2e}")
        
        Flat -= np.median(Flat[5:-5,5:-5])
        Flat = 10**Flat

        if show:
            fig, ax = plt.subplots(figsize=[6,3.5], sharex=True, sharey=True)
            im = ax.imshow(Flat, plt.cm.gray, origin='lower', interpolation='bilinear')
            im.set_clim(Flat[10:-10,10:-10].min(),Flat[10:-10,10:-10].max())
            ax.set_xlabel("Wavelength (pix)")
            ax.set_ylabel("Slit (pix)")
            ax.set_title("Flat Pattern")
            fig.tight_layout()
            fig.show()
            try:
                fig.canvas.manager.window.move(600,1080-350)
            except:
                    pass
            
        return Flat
    
    def saveFits(self, sdir=False, overwirte=False):
        """
        Save slit pattern and flat as a fits file

        Parameters
        ----------
        sdir: `str`, optional
            Save directory.
            If not, the data will be save at the '../proc/cal' directory for the input flat file.
        overwrite: `bool`, optional
            If `True`, overwrite the output file if it exists, Default is `False`

        Returns
        -------
        None
        """
        if not sdir:
            sdir = self.sdir
        if not isdir(sdir):
            makedirs(sdir)

        rfname = basename(self.fflat)
        tmp = rfname.replace('_Flat', '')
        fname = tmp.replace('FISS_', 'FISS_FLAT_')
        sname = tmp.replace('FISS_', 'FISS_SLIT_')
        fname = join(sdir, fname)
        sname = join(sdir, sname)

        if self.h['STRTIME'].find('.') < 10:
            self.h['STRTIME'] = self.h['STRTIME'].replace('-', 'T').replace('.', '-')
        if self.h['ENDTIME'].find('.') < 10:
            self.h['ENDTIME'] = self.h['ENDTIME'].replace('-', 'T').replace('.', '-')

        obstime = (Time(self.h['STRTIME']).jd + Time(self.h['ENDTIME']).jd)/2
        obstime = Time(obstime, format='jd').isot

        # save slit
        hdu = fits.PrimaryHDU(self.Slit)
        hdu.header['EXPTIME'] = (self.h['EXPTIME'], 'Second')
        hdu.header['OBSTIME'] = (obstime, 'Observation Time (UT)')
        hdu.header['DATE'] = (self.h['DATE'], 'File Creation Date (UT)')
        hdu.header['STRTIME'] = (self.h['STRTIME'], 'Scan Start Time')
        hdu.header['ENDTIME'] = (self.h['ENDTIME'], 'Scan Finish Time')
        hdu.header['TILT'] = (self.tilt, 'Degree')
        hdu.header['CCDNAME'] = (self.h['CCDNAME'], 'Prodctname of CCD')
        
        
        try:
            hdu.header['WAVELEN'] = (self.h['WAVELEN'], 'Angstrom')
        except:
            pass
        try:
            hdu.header['GRATWVLN'] = (self.h['GRATWVLN'], 'Angstrom')
        except:
            pass
        for comment in self.h['COMMENT']:
            hdu.header.add_history(comment)
        hdu.writeto(sname, overwrite=overwirte)

        # save flat file
        # save slit
        fhdu = fits.PrimaryHDU(self.Flat)
        fhdu.header['TILT'] = (self.tilt, 'Degree')
        fhdu.header['CCDNAME'] = (self.h['CCDNAME'], 'Prodctname of CCD')
        fhdu.header['EXPTIME'] = (self.h['EXPTIME'], 'Second')
        fhdu.header['STRTIME'] = (self.h['STRTIME'], 'Scan Start Time')
        fhdu.header['ENDTIME'] = (self.h['ENDTIME'], 'Scan Finish Time')
        try:
            fhdu.header['WAVELEN'] = (self.h['WAVELEN'], 'Angstrom')
        except:
            pass
        try:
            fhdu.header['GRATWVLN'] = (self.h['GRATWVLN'], 'Angstrom')
        except:
            pass
        for comment in self.h['COMMENT']:
            fhdu.header.add_history(comment)
        fhdu.header.add_history('Slit pattern subtracted')
        fhdu.writeto(fname, overwrite=overwirte)

    def slitTest(self, i=3):
        if self.tfig == None:
            rlogRF = rotImage(self.logRF, np.deg2rad(-self.tilt), cubic=True, missing=None)
            self.tmpo = 10**(rlogRF - rlogRF.mean(1)[:,None,:])
            self.tmps = 10**(self.tsi - self.tsi.mean(1)[:,None,:])
            m = self.tmps.mean()
            std = self.tmps.std()
            self.tfig, ax = plt.subplots(1, 2, figsize=[14,5], sharey=True, sharex=True)
            self.imtmpo = ax[0].imshow(self.tmpo[i], plt.cm.gray, origin='lower')
            self.imtmps = ax[1].imshow(self.tmps[i], plt.cm.gray, origin='lower')
            self.imtmpo.set_clim(m-std*0.5, m+std*0.5)
            self.imtmps.set_clim(m-std*0.5, m+std*0.5)
            ax[0].set_aspect(adjustable='box', aspect='auto')
            ax[1].set_aspect(adjustable='box', aspect='auto')
            ax[0].set_xlabel('Wavelength (pix)')
            ax[1].set_xlabel('Wavelength (pix)')
            ax[0].set_ylabel('Slit (pix)')
            ax[0].set_title('Original')
            ax[1].set_title('Shift correction')
            self.tfig.tight_layout()
            self.tfig.show()
        else:            
            self.imtmpo.set_data(self.tmpo[i])
            self.imtmps.set_data(self.tmps[i])
            self.tfig.canvas.draw_idle()
    
    def atlas_subtraction(self):
        wave, intensity = read_atlas()
        li = np.log10(intensity)
        fprof = np.median(self.logF[:,5:-5], axis=1)
        
        try:
            crval1 = float(self.h['GRATWVLN'])
        except:
            crval1 = float(self.h['WAVELEN'])

        if self.h['CCDNAME'] == 'DV897_BV': # cam A
            if abs(crval1 - 6562.817) < 3:
                crval1 = 6562.817
            elif abs(crval1 - 5889.95) < 3:
                crval1 = 5889.95
            elif abs(crval1 - 5875.618) < 3:
                crval1 = 5875.618
            res = get_echelle_res(crval1, 0.93)
        elif self.h['CCDNAME'] == 'DU8285_VP': # cam B
            if abs(crval1 - 8542.09) < 3:
                crval1 = 8542.09
            elif abs(crval1 - 6562.817) < 3:
                crval1 = 8542.09
            elif abs(crval1 - 5889.95) < 3:
                crval1 = 5434.5235
            res = get_echelle_res(crval1, 1.92)

        nf, nw = fprof.shape
        wr = spectral_range(res['alpha'], res['order'], nw)
        dw = (wr[1]-wr[0])/nw
        wv = np.arange(nw)*dw + wr[0]
        wh = (wave >= crval1-15) * (wave <= crval1+15)
        wave2 = wave[wh]
        smin = [wave2[0]]
        smax = [wave2[-1]]
        order = [len(wave2)]
        interp = CubicSpline(smin, smax, order, li[wh])
        refI = interp(wv[:,None])*np.ones((4,nw))

        self.tsh = np.zeros(nf)
        aprof = np.zeros((nf,nw))
        for i, prof in enumerate(fprof):
            d2p = np.gradient(np.gradient(prof))
            d2p = d2p*np.ones((4,nw))
            iteration = 0
            d2r = np.gradient(np.gradient(refI, axis=1), axis=1)
            # sh = alignOffset(d2r[:,5:-5], d2p[:,5:-5])
            sh = alignOffset(refI[:,5:-5], prof[5:-5]*np.ones((4,nw-10)))
            
            wvl = wv.copy()
            self.tsh[i] += sh[1,0]
            while abs(sh[1,0]) >= 1e-1:
                wvl += sh[1,0]*dw
                testI = interp(wvl[:,None])*np.ones((4,nw))
                d2r = np.gradient(np.gradient(testI, axis=1), axis=1)
                # sh = alignOffset(d2r[:, 5:-5], d2p[:, 5:-5])
                sh = alignOffset(testI[:,5:-5], prof[5:-5]*np.ones((4,nw-10)))
                
                self.tsh[i] += sh[1,0]
                iteration += 1
                if iteration == 10:
                    print(f"{i} break")
                    break
            wvl = wv.copy() + self.tsh[i]*dw
            aprof[i] = interp(wvl[:,None])


        self.mlf = self.logF[:,5:-5].mean(1)
        self.tap = aprof.copy()
        ap =  aprof
        self.lprof = lprof = np.median(self.logF[:,5:-5],1)
        rmin = ap[self.nf//2].argmin()
        A = ap[self.nf//2,rmin-10:rmin+10].min()
        B = ap[self.nf//2,-30:-10].mean()
        R = A-B
        a = lprof[self.nf//2,rmin-10:rmin+10].min()
        b = lprof[self.nf//2,-30:-10].mean()
        r = a-b
        ap *= r/R
        A1 = ap[self.nf//2,rmin-10:rmin+10].min()
        ap += a-A1
        # for i in range(self.nf):
        #     sh = int(self.tsh[i] - self.tsh[self.nf//2])
        #     # am1 = ap[i,rmin-sh-10:rmin-sh+10].min()
        #     # am2 = ap[i,-30:-10].mean()
        #     # ra = am2 - am1
        #     pm1 = lprof[i,rmin-sh-10:rmin-sh+10].min()
        #     # pm2 = ap[i,-30:-10].mean()
        #     # rp = pm2 - pm1
        #     # r = rp/ra
        #     ap[i] *= r
        #     ys = pm1 - ap[i,rmin-sh-10:rmin-sh+10].min()
        #     ap[i] += ys

        self.ap = ap
        self.rmFlat = self.logF - ap[:,None,:]
        self.rmFlat -= self.rmFlat[:,5:-5,5:-5].mean((1,2))[:,None,None]

    def get_spectrumMaskRange(self, data):
        prof = self.logF[:,5:-5,5:-5].mean(1)
        whmin = prof.argmin(1)
        tt = data[5:-5,5:-5].mean(0) - data[5:-5,5:-5].min()
        pars = [tt[whmin], whmin, 5]
        x = np.arange(tt.shape[1])
        cp, cr = curve_fit(Gaussian, x[whmin-5:whmin+5], tt[whmin-5:whmin+5], p0=pars)
        return [cp[1]-cp[2]*1.5,cp[1]+cp[2]*1.5]

def read_atlas():
    dirn = dirname(abspath(__file__))
    f = join(dirn, 'solar_atlas.npz')
    if not isfile(f):
        url = 'http://fiss.snu.ac.kr/static/atlas/solar_atlas.npz'
        urlretrieve(url, f)
    atlas = np.load(f)
    wave = atlas['wave']
    intensity = atlas['intensity']
    return wave, intensity

def readcal(pcdir):
    """
    Read FLAT ans SLIT pattern files.
    """
    lcam = ['A', 'B']
    res = {}

    for cam in lcam:
        lflat = glob(join(pcdir, f"FISS_FLAT*{cam}.fts"))
        lslit = glob(join(pcdir, f"FISS_SLIT*{cam}.fts"))
        lflat.sort()
        lslit.sort()
        h = fits.getheader(lflat[0])
        res[f'Tilt_{cam}'] = h['TILT']
        nf = len(lflat)
        jd = np.zeros(nf)
        flat = np.zeros((nf, h['naxis2'], h['naxis1']))
        slit = np.zeros((nf, h['naxis2'], h['naxis1']))
        for i, fflat in enumerate(lflat):
            oflat = fits.open(fflat)[0]
            flat[i] = oflat.data
            hf = oflat.header

            slit[i] = fits.getdata(lslit[i])
            jd[i] = Time(hf['obstime']).jd

        res[f'Flat_{cam}'] = flat
        res[f'Slit_{cam}'] = slit
        res[f'JD_{cam}'] = jd

    return res

def preprocess(f, outname, flat, slit, dark, tilt, curve_coeff, cent_wv=False, overwrite=False, ret=False):


    opn = fits.open(f)[0]
    h = opn.header
    data = opn.data - dark

    d2rd = np.gradient(np.gradient(data, axis=1), axis=1)
    d2rs = np.gradient(np.gradient(slit, axis=0), axis=0)
    wh = int(np.median(np.abs(d2rs[5:-5,5:-5]).argmax(0))) + 5
    ref = d2rs[wh-16:wh+16,5:-5]
    step = 10
    nx = len(d2rd[::step])
    shy = np.zeros(nx)
    for i, frd in enumerate(d2rd[::step]):
        spec = frd[wh-16:wh+16,5:-5]
        sh = alignOffset(spec, ref)
        shy[i] = sh[0,0]

    sh[0,0] = np.median(shy)
    sh[1,0] = 0
    smflat = shiftImage(slit*flat, sh, missing=None, cubic=True)
    data = data/smflat

    ti = tilt_correction(data, tilt)
    ci = curvature_correction(ti, curve_coeff)

    wvpar = wv_calib_atlas(ci, h, cent_wv)
    hdu = fits.PrimaryHDU(ci)
    hdu.header = h
    hdu.header['CRPIX1'] = (wvpar[0], 'reference pixel position')
    hdu.header['CRDELT1'] = (wvpar[1], 'angstrom/pixel')
    hdu.header['CRVAL1'] = (wvpar[2], 'reference wavelength (angstrom)')
    hdu.header['WAVELEN'] = (f"{wvpar[2]:.2f}", 'reference wavelength')
    hdu.header['TILT'] = (tilt, 'tilt angle in degree')
    hdu.header.add_history('Bias+Dark subtracted')
    hdu.header.add_history('Slit pattern subtracted')
    hdu.header.add_history('Flat fileded')
    hdu.header.add_history(f'{tilt:.3f} degree tilt corrected')
    hdu.header.add_history('Curvature corrected')
    hdu.writeto(outname, overwrite=overwrite)

    if ret:
        return ci, hdu.header

def wv_calib_atlas(data, header, cent_wv=False):
    wave, intensity = read_atlas()

    data1 = np.median(data, axis=0) if data.ndim == 3 else data

    if cent_wv:
        crval1 = cent_wv
    else:
        try:
            crval1 = float(header['GRATWVLN'])
        except:
            crval1 = float(header['WAVELEN'])

    if header['CCDNAME'] == 'DV897_BV': # cam A
        if abs(crval1 - 6562.817) < 3:
            crval1 = 6562.817
        elif abs(crval1 - 5889.95) < 3:
            crval1 = 5889.95
        elif abs(crval1 - 5875.618) < 3:
            crval1 = 5875.618
        res = get_echelle_res(crval1, 0.93)
    elif header['CCDNAME'] == 'DU8285_VP': # cam B
        if abs(crval1 - 8542.09) < 3:
            crval1 = 8542.09
        elif abs(crval1 - 6562.817) < 3:
            crval1 = 8542.09
        elif abs(crval1 - 5889.95) < 3:
            crval1 = 5434.5235
        res = get_echelle_res(crval1, 1.92)

    data1 = data1[10:-10]
    ny, nw = data1.shape
    wr = spectral_range(res['alpha'], res['order'], nw)
    dw = (wr[1]-wr[0])/nw
    wv = np.arange(nw)*dw + wr[0]
    wh = (wave >= crval1-10) * (wave <= crval1+10)
    wave2 = wave[wh]
    smin = [wave2[0]]
    smax = [wave2[-1]]
    order = [len(wave2)]
    interp = CubicSpline(smin, smax, order, intensity[wh])
    ii = interp(wv[:,None])

    wpix = np.arange(nw)
    cpix = np.zeros(ny)
    refI = ii*np.ones((4,nw))
    for i, prof in enumerate(data1):
        prof = prof *np.ones((4,nw))
        wmax = 0
        wsh = 10
        iteration = 0
        while abs(wsh) >= 1e-1:
            if wmax == 0:
                sh = alignOffset(prof, refI)
            elif wmax >= 1:
                sh = alignOffset(prof[:,:-int(wmax)], refI[:,:-int(wmax)])
            else:
                sh = alignOffset(prof[:,-int(wmax):], refI[:,-int(wmax):])
            prof = shiftImage(prof, -sh, missing=None, cubic=True)
            wsh = sh[-1][0]
            wmax += wsh
            iteration += 1
            if iteration == 10:
                print(wsh)
                break
        wv_cor = wv - wmax*dw
        mmin = [wv_cor[0]]
        mmax = [wv_cor[-1]]
        order = [nw]
        cinterp = CubicSpline(mmin, mmax, order, wpix)
        cpix[i] = cinterp(np.array([[crval1],[crval1]]))[0]
    cpix1 = np.median(cpix)
    wvpar = [cpix1, dw, crval1]

    return wvpar
        
def get_echelle_res(lamb, theta=0.93, grooves=79, blazeAngle=63.4):
    """
    Determine the parameters of an echelle grating.

    Parameters
    ----------
    lamb: `float`
        Target wavelength (Angstrom)
    theta: `float` (optional)
        Deflection angle (angle between incidence and reflection) in degree
    grooves: `int` (optional)
        Number of grooves per mm
    blazeAngle: `float` (optional)
        Blaze angle (degree)

    Returns
    -------
    res: `dict`
        'alpha' - Incident angle
        'order' - Grating order of the peak brightness
        'brightness' - Relative brightness for given wavelength.
    """
    phi = np.deg2rad(blazeAngle)
    sigma = 1e7/grooves # unit of angstrom
    m0 = np.round(2*sigma/lamb*np.sin(phi))
    marr = np.arange(-2,3) + m0
    
    B = np.zeros(5)
    Dalpha = np.zeros(5)
    for i, m in enumerate(marr):
        alpha = phi
        for j in range(10):
            f = np.sin(alpha) + np.sin(alpha-np.deg2rad(theta))-m*lamb/sigma
            df = np.cos(alpha) + np.cos(alpha-np.deg2rad(theta))
            alpha -= f/df
            # print(f"m: {m}, alpha: {alpha}")

        Dalpha[i] = np.rad2deg(alpha)
        beta = alpha - np.deg2rad(theta)
        x = np.pi*sigma*np.cos(phi)/lamb*(np.sin(alpha-phi)+np.sin(beta-phi))
        B[i] = 1 if x == 0 else (np.sin(x)/x)**2

    m = marr[B.argmax()]
    DA = Dalpha[B.argmax()]
    res = {"alpha": DA,
           "order": int(m),
           "brightness": B.max()
           }
    
    return res

def spectral_range(alpha, order, nw=512):
    f = 1.5 # FISS collimator focal length in m
    grooves = 79
    sigma = 1e7 / grooves
    
    if nw % 512 == 0:
        theta = np.deg2rad(0.93)
        domain = np.array([0.5, -0.5])
    elif nw % 502 ==0:
        theta = np.deg2rad(1.92)
        domain = np.array([-0.503, 0.503])
    else:
        raise ValueError(f"nw should be either 512 or 502, not {nw}")
    width = nw*16e-6 # Detector width in m
    theta_range = theta + domain*width/f
    wl = sigma/order * (np.sin(np.deg2rad(alpha)) + np.sin(np.deg2rad(alpha) - theta_range))

    return wl

def PCA_compression(fproc, Evec=None, pfile=None, ncoeff=None, tol=1e-1, ret=False):
    """
    Compress the given 3D data by using the PCA analysis as described in Chae et al. 2013.
    
    Parameters
    ----------
    frpoc: `str`
        Processed file to be compressed.
    Evec: `~numpy.array` (optional)
        Eigen vector of the pfile. It will be used to compress the data for a given pfile data. When you make the pfile, do not give this value!
    pfile: `str` (optional)
        pfile (eigen vector file) of the PCA compression. If this and Evec is not given, calculate the eigen vector from the given fproc file.
    ncoeff: `int` (optional)
        The number of coefficients of the principal components. If it is not given automatically find the ncoeff until the eigenvalue is higher than the given tolerance value. Min=30, Max=50
    tol: `float` (optional)
        The tolerance to find the ncoeff eigenvalue. The number of the coefficients is equals the number of the componenet that has an eigenvalue of higher than the tolerance.
        Default is 1e-1.
    ret: `bool`
        If true return the Eigenvector, original data and compressed data and Eignevalue.
        Default is False.

    """
    opn = fits.open(fproc)[0]
    h = opn.header
    data = opn.data.astype(float)
    odata = data.copy()

    nx, ny, nw = data.shape
    Eval = None

    if pfile is None and Evec is None:
        pfile = fproc.replace('.fts', '_p.fts')
        pfile = pfile.replace('proc', 'comp')
        dirn = dirname(pfile)
        if not isdir(dirn):
            makedirs(dirn)
            

        ranx = np.random.uniform(0, nx-1, 5000).round().astype(int)
        rany = np.random.uniform(0, ny-1, 5000).round().astype(int)
        tmp = data[ranx, rany]
        tmpMin = tmp.min(1)
        wh = tmpMin > 1
        nd = wh.sum()
        if nd >= 2000:
            spgr = tmp[wh][:2000]
        else:
            spgr = tmp[wh]
        
        spgr /= spgr.mean(1)[:,None]
        c_arr = spgr.T.dot(spgr)
        m = c_arr.mean()
        c_arr = np.nan_to_num(c_arr, True, m,m,m)
        # c_arr[c_arr == -np.inf] = 0
        # c_arr[c_arr == np.inf] = 0

        Eval, Evec = np.linalg.eig(c_arr)
        if ncoeff is None:
            NC = (Eval >= tol).sum()
            NC = NC if NC < 50 else 50
            NC = NC if NC > 30 else 30
            print(f"eigenvalue[{NC}]: {Eval[50]:.3f}")
        else:
            NC = ncoeff
        Evec = Evec[:,:NC].T

        hdu = fits.PrimaryHDU(Evec.astype('float32'))
        hdu.header['NC'] = NC
        hdu.header['date'] = h['date']
        hdu.writeto(pfile, overwrite=True)
        
        
    else:
        if Evec is None:
            opn = fits.open(pfile)
            ph = opn.header
            Evec = opn.data

        NC = Evec.shape[0]


    coeff = np.zeros((nx, ny, NC+1))
    cfile = fproc.replace('.fts', '_c.fts')
    cfile = cfile.replace('proc', 'comp')
    wh = data < 1
    data[wh] = 1
    av = data.mean(2)
    data /= av[:,:,None]
    coeff[:,:,NC] = np.log10(av)
    for i in range(NC):
        coeff[:,:,i] = (data[:,:,:]*Evec[i]).sum(2)

    bscale = np.abs(coeff).max()/2e4
    coeff = np.round(coeff/bscale)


    hdu = fits.PrimaryHDU(coeff.astype('int16'))
    hdu.header["bscale"] = bscale
    hdu.header['pfile'] = basename(pfile)
    for cards in h.cards:
        hdu.header.add_comment(f"{cards[0]} = {cards[1]} / {cards[2]}")

    hdu.writeto(cfile, overwrite=True)

    if ret and pfile is not None:
        c = coeff*bscale
        spec = c[:,:,:NC].dot(Evec)
        spec *= 10**c[:,:,-1][:,:,None]
        if Eval is None:
            return Evec, spec, odata
        else:
            return Evec, spec, odata, Eval[NC]

def yf2sp(yf, tolRate=1):
    """
    Calculate the slit pattern using the y-directional Fringe pattern
    """
    d2y = np.gradient(np.gradient(yf,axis=0), axis=0).mean(1)
    pks = find_peaks(d2y[5:-5], d2y[5:-5].std()*tolRate)[0]+5
    myf = data_mask_and_fill(yf, [pks-1,pks+2], axis=0, kind='slinear')
    s1 = myf[:,5:-5].mean(1)[:,None]
    sp = yf-myf+s1
    fsp = fft(sp, axis=1)
    fsp[:,10:-9] = 0
    sp = ifft(fsp,axis=1).real
    return sp


def raw2sp(raw, pks):
    mr = np.log10(raw.mean(0))
    mr = np.nan_to_num(mr, True, 1,1,1)
    mRaw = mr - mr[5:-5].mean(0)
    ft = fft(mRaw, axis=0)
    ft[:10] = 0
    ft[-9:] = 0
    k = ifft(ft, axis=0).real
    mk = np.median(k[:, 5:-5], 1)
    yy = np.arange(mRaw.shape[0])
    x = mk[pks-2]
    y = yy[pks-2]
    for i in range(-1,3):
        x = np.append(x,mk[pks+i])
        y = np.append(y,yy[pks+i])
    idx = y[x.argmin()]
    xx = np.arange(mRaw.shape[1])
    coef = np.polyfit(xx[5:-5], k[idx,5:-5],1)
    kkf = np.polyval(coef, xx)
    tmp = k[idx,5:-5] - kkf[5:-5]
    val = tmp.mean()+tmp.std()
    wh = tmp > val
    xx = np.arange(mRaw.shape[1]-10)
    ct = xx[len(xx)//2]+5
    hw = int(len(xx[wh])*1.5 // 2)
    mm = ct-hw
    mM = ct+hw
    tt = data_mask_and_fill(k, [[mm],[mM]], axis=1, kind='slinear')

    mskTT = data_mask_and_fill(tt, [pks-2,pks+3], axis=0, kind='slinear')
    sp = tt-mskTT
    fsp = fft(sp, axis=1)
    fsp[:,10:-9] = 0
    sp = ifft(fsp,axis=1).real
    sp -= sp[5:-5,5:-5].mean()
    return 10**sp


def rawYF(sraw, aws):
    mr = np.log10(sraw.mean(0))
    mr = np.nan_to_num(mr, True, 1,1,1)
    mRaw = mr - mr[5:-5].mean(0) - mr[:, 5:-5].mean(1)[:, None]
    wvlet = Wavelet(mRaw, dt=1, axis=0)
    freq = np.arctan2(wvlet.wavelet.imag, wvlet.wavelet.real)
    wvs = aws*(np.cos(freq) + 1j*np.sin(freq))
    fringe = wvlet.iwavelet(wvs, wvlet.scale)
    fringe -= fringe[5:-5,5:-5].mean()
    return 10**fringe.T

def YFart_correction(yf):
    syf = np.zeros(yf.shape)
    sp = np.zeros(yf.shape)
    for i, el in enumerate(yf):
        sp[i] = yf2sp(el)
        syf[i] = el - sp[i]
        wvl = Wavelet(syf[i], dt=1, axis=0)
        if i == 0:
            s1,s2,s3 = wvl.wavelet.shape
            aws = np.zeros((yf.shape[0], s1, s2, s3))
            phase = np.zeros((yf.shape[0], s1, s2, s3))
            spec = np.zeros((yf.shape[0], s1, s2, s3), dtype=complex)
        aws[i] = np.abs(wvl.wavelet)
        spec[i] = wvl.wavelet
        phase[i] = np.arctan2(wvl.wavelet.imag, wvl.wavelet.real)
    aws = np.abs(spec.mean(0))
    
    for i in range(yf.shape[0]):
        syf[i] = wvl.iwavelet(aws*(np.cos(phase[i]) + 1j*np.sin(phase[i])), wvl.scale).T
    return syf + sp

def calShift(raw, sp, pks):
    lsp = np.log10(sp)
    rd2y = np.gradient(np.gradient(lsp,axis=0), axis=0)
    wp = 40
    npks = len(pks)
    lraw = np.log10(raw)
    m = np.log10(raw[:,5:-5,5:-5].mean())
    m = np.nan_to_num(m, True, 1,1,1)
    lraw = np.nan_to_num(lraw, True, m, m, m)
    data = lraw.mean(0) - lraw[:,5:-5].mean((0,1))
    d2y = np.gradient(np.gradient(data,axis=0), axis=0)

    sh = 0
    ash = np.zeros(npks)
    if pks[0] < 16:
        ss = 1
    else:
        ss = 0
    if pks[-1] > d2y.shape[0]-16:
        ee = len(pks)-1
    else:
        ee = len(pks) 
    for i, whd in enumerate(pks[ss:ee]):
        rimg = rd2y[whd-8:whd+8, 10:-10]
        img = d2y[whd-8:whd+8, 10:-10]
        ash[i] = alignOffset(img, rimg)[0,0]
    sh = np.median(ash)
    s = np.zeros((2,1))
    mx = int(np.round(sh))
    s[0,0] = mx
    
    ssp = shiftImage(sp, s, missing=sp[5:-5,5:-5].mean())

    tpks = find_peaks(d2y[5:-5,10], d2y[5:-5,10].std())[0]+5
    spks = pks+mx
    if mx < 0:
        pp = tpks[tpks >= d2y.shape[0]+mx-5]
    else:
        pp = tpks[tpks <= mx+5]
    if pp.size:
        spks = np.append(spks, pp)
    
    return ssp, spks

