import numpy as np
from astropy.io import fits
from interpolation.splines import LinearSpline, CubicSpline
from fisspy.image.base import alignoffset, rot
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from os.path import join, isdir, dirname, basename
from os import getcwd, makedirs

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

    dy_img = np.gradient(img, axis=0)
    whd = np.abs(dy_img[20:-20,wp:wp+20].mean(1)).argmax() + 20
    i1 = dy_img[whd-16:whd+16, wp:wp+16]
    i2 = dy_img[whd-16:whd+16, -(wp+16):-wp]
    shift = alignoffset(i2, i1)
    if tilt is None:
        Tilt = np.rad2deg(np.arctan2(shift[0], nw - wp*2))[0]
    else:
        Tilt = tilt


    if show:
        rimg = rot(img, np.deg2rad(-Tilt), cubic=True, missing=-1)
        
        fig, ax = plt.subplots(2,1, figsize=[8, 8], sharey=True, sharex=True, num='Get_tilt')
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

def get_curve_par(cflat, tilt, show=False):
    """
    Calculate the curvature of the spectrum by applying the second order polynominal fit.

    Parameters
    ----------
    cflat: `~numpy.array`
        2-dimensional flat corrected raw flat image. ex) cflat = rawflat[3]/Flat/Slit.
    tilt: `float`
        Tilt angle of the image in degree unit.
    show: `bool`, optional
        If `True`, draw wavelength shift along the slit direction and fitting. Default is `False`

    Returns
    -------
    p: `list`
        Results of the numpy.polyfit of the second-order polynomial fitting.
    """
    ny, nw = cflat.shape
    rf = rot(cflat, np.deg2rad(-tilt), missing=-1)
    d2flat = np.gradient(np.gradient(rf[:,5:-5], axis=1), axis=1)
    d2flat = savgol_filter(d2flat, 40, 2, axis=1)

    dw = np.zeros(ny)
    one = np.ones((4,nw-10))
    # if method == 0:
    #     ref = d2flat[ny//2-2:ny//2+2].mean(1)[:,None] * one
    #     for i, prof in enumerate(d2flat):
    #         dw[i] = alignoffset(prof*one, ref)[1]
    
    
    for direction in range(-1,2,2):
        prof0 = d2flat[ny//2]
        for i, prof in enumerate(d2flat[ny//2 + direction::direction]):
            dw[direction*(i+1) + ny//2] = alignoffset(prof*one, prof0*one)[1]
            dw[direction*(i+1) + ny//2] += dw[direction*(i) + ny//2]
            prof0 = prof

    y = np.arange(ny)
    p = np.polyfit(y, dw, 2)

    if show:
        fig, ax = plt.subplots(figsize=[8,5], num='curvature parameter')
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
        ax.legend()
        fig.tight_layout()
        fig.show()

    return p

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
    ti = rot(img, np.deg2rad(-tilt), cubic=cubic, missing=-1)

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
        inp[i] = tmp
        idx[i] = sh//2
    inp[-1] = w.reshape(size)
    inp[-2] = y.reshape(size)

    ccImg = interp(inp.T).reshape(shape)

    if show:
        fig, ax = plt.subplots(1,2, figsize=[8,8], sharex=True, sharey=True, num='Curvature corrected')
        oimg = img[tuple(idx)].squeeze()
        prof = oimg[20]
        dp2 = np.gradient(np.gradient(prof))
        wh = dp2.argmax()
        cimg = ccImg[tuple(idx)].squeeze()
        oim = ax[0].imshow(oimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        cim = ax[1].imshow(cimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        m = oimg[5:-5,wh-10:wh+10].mean()
        std = oimg[5:-5,wh-10:wh+10].std()
        oim.set_clim(m-std*1.5, m+std*1.5)
        cim.set_clim(m-std*1.5, m+std*1.5)
        ax[0].set_xlim(wh-10,wh+10)
        ax[0].set_aspect(adjustable='box', aspect='auto')
        ax[1].set_aspect(adjustable='box', aspect='auto')
        ax[0].set_xlabel('Wavelength (pix)')
        ax[1].set_xlabel('Wavelength (pix)')
        ax[0].set_ylabel('Slit (pix)')
        ax[0].set_title('Original')
        ax[1].set_title('Curvature corrected')
        fig.tight_layout()
        fig.show()
    return ccImg
    

class calFlat:
    def __init__(self, fflat, ffoc=None, tilt=None, autorun=True, save=True, show=False, maxiter=10, msk=None):
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
        
        fdir = dirname(fflat)
        if not fdir:
            fdir = getcwd()

        self.sdir = join(dirname(fdir), 'proc', 'cal')
        # get tilt angle in degree
        if ffoc is not None:
            foc = fits.getdata(ffoc)
            self.mfoc = foc.mean(0)
            self.tilt = get_tilt(self.mfoc, show=show)

        else:
            self.tilt = get_tilt(10**self.mlogRF, tilt=tilt, show=show)



        print(f"Tilt: {self.tilt:.2f} degree")

        
        if autorun:
            # get slit pattern
            self.Slit = self.make_slit_pattern(cubic=True, show=show)
            self.logSlit = np.log10(self.Slit)
            # remove the slit pattern
            self.logF = self.logRF - self.logSlit
            plt.pause(0.1)
            # save slit pattern
            self.logSlit -= np.median(self.logSlit)
            # get flat image
            self.Flat = self.gain_calib(maxiter=maxiter, msk=msk, show=show)
            self.cFlat = 10**(self.logRF[3] - np.log10(self.Flat) - self.logSlit)
            if save:
                self.saveFits(self.sdir)
        
            
            if show:
                fig, ax = plt.subplots(2, figsize=[9,9], num=f'Flat field {self.date}', sharex=True, sharey=True)
                im0 = ax[0].imshow(self.logRF[3], plt.cm.gray, origin='lower', interpolation='bilinear')
                m = self.logRF[3].mean()
                std = self.logRF[3].std()
                im0.set_clim(m-std, m+std)
                ax[0].set_ylabel("Slit (pix)")
                ax[0].set_title("Raw Data")
                logf = np.log10(self.Flat)
                corI = self.logRF[3] - logf - self.logSlit
                im = ax[1].imshow(corI, plt.cm.gray, origin='lower', interpolation='bilinear')
                # im.set_clim(corI[10:-10,10:-10].min(),corI[10:-10,10:-10].max())
                m = corI[10:-10,10:-10].mean()
                std = corI[10:-10,10:-10].std()
                self.im = im
                im.set_clim(m-std, m+std)
                ax[1].set_xlabel("Wavelength (pix)")
                ax[1].set_ylabel("Slit (pix)")
                ax[1].set_title("Flat correction")
                fig.tight_layout()
                fig.show()

    def make_slit_pattern(self, cubic=True, show=False):
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
        ri = rot(self.mlogRF, np.deg2rad(-self.tilt), cubic=cubic, missing=-1)

        # rslit = ri[:,40:-40].mean(1)[:,None] * np.ones([self.ny,self.nw])
        rslit = np.median(ri[:,40:-40], axis=1)[:,None] * np.ones([self.ny,self.nw])
        logSlit = rot(rslit, np.deg2rad(self.tilt), cubic=cubic, missing=-1)

        if show:
            fig, ax = plt.subplots(figsize=[9,5], num=f'Slit Pattern {self.date}')
            im = ax.imshow(logSlit, plt.cm.gray, origin='lower', interpolation='bilinear')
            ax.set_xlabel("Wavelength (pix)")
            ax.set_ylabel("Slit (pix)")
            ax.set_title(f"Slit Pattern")
            fig.tight_layout()
            fig.show()
            
        return 10**logSlit

    def gain_calib(self, maxiter=10, msk=None, show=False):
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
        window_length = 40
        polyorder = 2
        deriv = 0
        delta = 1.0
        mode = 'interp'
        cval = 0.0

        
        if msk is None:
            self.der2 = np.gradient(np.gradient(self.logF, axis=2), axis=2)
            self.der2 -= self.der2[:,10:-10,10:-10].mean((1,2))[:,None, None]
            std = self.der2[:,10:-10,10:-10].std((1,2))[:,None,None]
            msk = np.exp(-0.5*np.abs((self.der2/std))**2)
            msk = savgol_filter(msk, window_length, polyorder,
                      deriv= deriv, delta= delta, cval= cval,
                      mode= mode, axis=2)
        self.msk = msk
        self.C = (self.logF*msk).sum((1,2))/msk.sum((1,2))
        self.C -= self.C.mean()

        Flat = np.median(self.logF, axis=0)
        Flat -= np.median(Flat)
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
            self.xi[k] = xdum[self.logF[k, hy] == self.logF[k, hy, 5:-5].min()][0]

        for k in range(self.nf-1):
            img1 = (self.logF[k+1] - Flat)[hy-10:hy+10].mean(0)*one
            img2 = (self.logF[k] - Flat)[hy-10:hy+10].mean(0)*one
            sh = alignoffset(img1, img2)
            dx = int(np.round(sh[1]))
            if dx < 0:
                img1 = (self.logF[k+1] - Flat)[hy-10:hy+10, :dx].mean(0)*one[:,:dx]
                img2 = (self.logF[k] - Flat)[hy-10:hy+10, -dx:].mean(0)*one[:,-dx:]
                sh, cor = alignoffset(img1, img2, cor=True)
            else:
                img1 = (self.logF[k+1] - Flat)[hy-10:hy+10, dx:].mean(0)*one[:,dx:]
                img2 = (self.logF[k] - Flat)[hy-10:hy+10, :-dx].mean(0)*one[:,:-dx]
                sh, cor = alignoffset(img1, img2, cor=True)
            self.x[k+1] = self.x[k] + sh[1] + dx
            print(f"k: {k+1}, x={self.x[k+1]}, cor={cor}")
        self.x -= np.median(self.x)


        self.dx = np.zeros([self.nf, self.ny])
        y = np.arange(self.ny)
        for k in range(self.nf):
            self.ref = np.gradient(np.gradient((self.logF[k]-Flat)[hy-10:hy+10].mean(0), axis=0), axis=0)*one
            for j in range(self.ny):
                img = np.gradient(np.gradient((self.logF[k] - Flat)[j], axis=0), axis=0)*one
                sh = alignoffset(img[:,5:-5], self.ref[:,5:-5])
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
        data = self.logF - Flat
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
            ob = (self.C[:,None,None] + obj1 + Flat - self.logF)*weight
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
            data = self.logF - Flat
            interp = CubicSpline(smin, smax, order, data)
            a = (self.C[:, None, None] + self.Object[None,None,:] - interp(inp.T).reshape(shape))*weight

            b = weight.sum((0,1))
            b[b < 1] = 1
            DelObject = - a.sum((0,1))/b
            self.Object += DelObject

            err = np.abs(DelFlat).max()
            print(f"iteration={i}, err: {err:.2e}")
        
        Flat -= np.median(Flat)
        Flat = 10**Flat

        if show:
            fig, ax = plt.subplots(figsize=[9,5], num=f'Flat Pattern {self.date}', sharex=True, sharey=True)
            im = ax.imshow(Flat, plt.cm.gray, origin='lower', interpolation='bilinear')
            im.set_clim(Flat[10:-10,10:-10].min(),Flat[10:-10,10:-10].max())
            ax.set_xlabel("Wavelength (pix)")
            ax.set_ylabel("Slit (pix)")
            ax.set_title("Flat Pattern")
            fig.tight_layout()
            fig.show()
            
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

        # save slit
        hdu = fits.PrimaryHDU(self.Slit)
        hdu.header['TILT'] = self.tilt
        hdu.header['CCDNAME'] = self.h['CCDNAME']
        hdu.header['EXPTIME'] = self.h['EXPTIME']
        hdu.header['STRTIME'] = self.h['STRTIME']
        hdu.header['ENDTIME'] = self.h['ENDTIME']
        try:
            hdu.header['WAVELEN'] = self.h['WAVELEN']
        except:
            pass
        try:
            hdu.header['GRATWVLN'] = self.h['GRATWVLN']
        except:
            pass
        for comment in self.h['COMMENT']:
            hdu.header.add_history(comment)
        hdu.writeto(sname, overwrite=overwirte)

        # save flat file
        # save slit
        fhdu = fits.PrimaryHDU(self.Flat)
        fhdu.header['TILT'] = self.tilt
        fhdu.header['CCDNAME'] = self.h['CCDNAME']
        fhdu.header['EXPTIME'] = self.h['EXPTIME']
        fhdu.header['STRTIME'] = self.h['STRTIME']
        fhdu.header['ENDTIME'] = self.h['ENDTIME']
        try:
            fhdu.header['WAVELEN'] = self.h['WAVELEN']
        except:
            pass
        try:
            fhdu.header['GRATWVLN'] = self.h['GRATWVLN']
        except:
            pass
        for comment in self.h['COMMENT']:
            fhdu.header.add_history(comment)
        fhdu.header.add_history('slit pattern subtracted')
        fhdu.writeto(fname, overwrite=overwirte)
        