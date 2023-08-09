import numpy as np
from astropy.io import fits
from interpolation.splines import LinearSpline, CubicSpline
from fisspy.image.base import alignoffset, rot
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def get_tilt(img, tilt=None, show=False):
    """
    Get a tilt angle of the spectrum camera in the unit of degree.

    Parameters
    ----------
    img : `~numpy.array`
        A two-dimensional `numpy.array` of the form ``(y, x)``.
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
        
        fig, ax = plt.subplots(1,2, figsize=[14, 4], sharey=True, sharex=True, num='Get_tilt')
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

        ax[0].set_xlabel('Wavelength (pix)')
        ax[1].set_xlabel('Wavelength (pix)')
        ax[0].set_ylabel('Y (pix)')
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

# make class (usful for debugging but should be optimized for memory)
class calFlat:
    def __init__(self, fflat, ffoc=None, tilt=None, autorun=True, save=False, show=False, maxiter=10, msk=None):
        opn = fits.open(fflat)[0]
        self.rawFlat = opn.data
        self.date = opn.header['date']
        self.nf, self.ny, self.nw =  self.rawFlat.shape
        self.logRF = np.log10(self.rawFlat)
        self.mlogRF = self.logRF.mean(0)
        self.tilt = tilt
        self.ffoc = ffoc
        
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
            self.logSlit = self.make_slit_pattern(cubic=True, show=show)
            # remove the slit pattern
            self.logF = self.logRF - self.logSlit
            plt.pause(0.1)
            # save slit pattern
            self.logSlit -= np.median(self.logSlit)
            # get flat image
            self.gain_calib(maxiter=maxiter, msk=msk, show=show)
        
            
            if show:
                fig, ax = plt.subplots(2, figsize=[9,9], num=f'Flat field {self.date}', sharex=True, sharey=True)
                im0 = ax[0].imshow(self.logRF[3], plt.cm.gray, origin='lower', interpolation='bilinear')
                m = self.logRF[3].mean()
                std = self.logRF[3].std()
                im0.set_clim(m-std, m+std)
                ax[0].set_ylabel("Y (pix)")
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
                ax[1].set_ylabel("Y (pix)")
                ax[1].set_title("Flat correction")
                fig.tight_layout()
                fig.show()

    def make_slit_pattern(self, cubic=True, show=False):
        """
        Make a slit pattern image for a given tilt angle

        """
        ri = rot(self.mlogRF, np.deg2rad(-self.tilt), cubic=cubic, missing=-1)

        # rslit = ri[:,40:-40].mean(1)[:,None] * np.ones([self.ny,self.nw])
        rslit = np.median(ri[:,40:-40], axis=1)[:,None] * np.ones([self.ny,self.nw])
        Slit = rot(rslit, np.deg2rad(self.tilt), cubic=cubic, missing=-1)

        if show:
            fig, ax = plt.subplots(figsize=[9,5], num=f'Slit Pattern {self.date}')
            im = ax.imshow(Slit, plt.cm.gray, origin='lower', interpolation='bilinear')
            ax.set_xlabel("Wavelength (pix)")
            ax.set_ylabel("Y (pix)")
            ax.set_title(f"Slit Pattern")
            fig.tight_layout()
            fig.show()

        # TODO save Fits 
            
        return Slit


    def gain_calib(self, maxiter=10, msk=None, show=False):
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

        self.Flat = np.median(self.logF, axis=0)
        self.Flat -= np.median(self.Flat)
        f1d = np.gradient(self.Flat, axis=1)
        f2d = np.gradient(f1d, axis=1)
        mask = (np.abs(f2d) <= f2d.std()) * (np.abs(f1d) <= f1d.std())
        mask[:,100:-100] = False

        w = np.arange(self.nw)
        for i, m in enumerate(mask):
            coeff = np.polyfit(w[m], self.Flat[i,m], 2)
            self.Flat[i] = np.polyval(coeff, w)

        self.x = np.zeros(self.nf)
        self.xi = np.zeros(self.nf)
        xdum = np.arange(self.nw, dtype=int)
        hy = int(self.ny//2)
        one = np.ones((4, self.nw))

        for k in range(self.nf):
            self.xi[k] = xdum[self.logF[k, hy] == self.logF[k, hy, 5:-5].min()][0]

        for k in range(self.nf-1):
            img1 = (self.logF[k+1] - self.Flat)[hy-10:hy+10].mean(0)*one
            img2 = (self.logF[k] - self.Flat)[hy-10:hy+10].mean(0)*one
            sh = alignoffset(img1, img2)
            dx = int(np.round(sh[1]))
            if dx < 0:
                img1 = (self.logF[k+1] - self.Flat)[hy-10:hy+10, :dx].mean(0)*one[:,:dx]
                img2 = (self.logF[k] - self.Flat)[hy-10:hy+10, -dx:].mean(0)*one[:,-dx:]
                sh, cor = alignoffset(img1, img2, cor=True)
            else:
                img1 = (self.logF[k+1] - self.Flat)[hy-10:hy+10, dx:].mean(0)*one[:,dx:]
                img2 = (self.logF[k] - self.Flat)[hy-10:hy+10, :-dx].mean(0)*one[:,:-dx]
                sh, cor = alignoffset(img1, img2, cor=True)
            self.x[k+1] = self.x[k] + sh[1] + dx
            print(f"k: {k+1}, x={self.x[k+1]}, cor={cor}")
        self.x -= np.median(self.x)


        self.dx = np.zeros([self.nf, self.ny])
        y = np.arange(self.ny)
        for k in range(self.nf):
            self.ref = np.gradient(np.gradient((self.logF[k]-self.Flat)[hy-10:hy+10].mean(0), axis=0), axis=0)*one
            for j in range(self.ny):
                img = np.gradient(np.gradient((self.logF[k] - self.Flat)[j], axis=0), axis=0)*one
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
        data = self.logF - self.Flat
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
            ob = (self.C[:,None,None] + obj1 + self.Flat - self.logF)*weight
            self.C -= ob.sum((1,2))/weight.sum((1,2))
            data = np.gradient(self.Object, axis=0)*ones
            interp = LinearSpline(smin, smax, order, data)
            oi = -interp(inp.T).reshape(shape)
            self.x -= (ob*oi).sum((1,2))/(weight*oi**2).sum((1,2))
            b = weight.sum(0)
            b[b < 1] = 1
            DelFlat = -ob.sum(0)/b
            self.Flat += DelFlat
            # self.Flat = savgol_filter(self.Flat, 20, polyorder,
            #           deriv= deriv, delta= delta, cval= cval,
            #           mode= mode, axis=1)

            pos = np.arange(self.nw)[None,None,:] + self.x[:,None,None] + self.dx[:,:,None]
            weight = (pos >= 0) * (pos < self.nw)
            pos[pos < 0] = 0
            pos[pos > self.nw-1] = self.nw-1
            interp = LinearSpline(smin, smax, order, self.msk)
            inp = np.array((f.reshape(size), y.reshape(size), pos.reshape(size)))
            weight = weight*interp(inp.T).reshape(shape)
            data = self.logF - self.Flat
            interp = CubicSpline(smin, smax, order, data)
            a = (self.C[:, None, None] + self.Object[None,None,:] - interp(inp.T).reshape(shape))*weight

            b = weight.sum((0,1))
            b[b < 1] = 1
            DelObject = - a.sum((0,1))/b
            self.Object += DelObject

            err = np.abs(DelFlat).max()
            print(f"iteration={i}, err: {err:.2e}")
        
        self.Flat -= np.median(self.Flat)
        self.Flat = 10**self.Flat

        if show:
            fig, ax = plt.subplots(figsize=[9,5], num=f'Flat Pattern {self.date}', sharex=True, sharey=True)
            im = ax.imshow(self.Flat, plt.cm.gray, origin='lower', interpolation='bilinear')
            im.set_clim(self.Flat[10:-10,10:-10].min(),self.Flat[10:-10,10:-10].max())
            ax.set_xlabel("Wavelength (pix)")
            ax.set_ylabel("Y (pix)")
            ax.set_title("Flat Pattern")
            fig.tight_layout()
            fig.show()