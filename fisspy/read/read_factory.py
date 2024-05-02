from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.constants as ac
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import ticker
from .. import cm
from .readbase import getRaster, getHeader, readFrame
from ..analysis import lambdameter
from ..image.interactive_image import singleBand
from ..correction import lineName, wvCalib, smoothingProf, corSLA
from ..analysis import FourierFilter, Wavelet, makeTDmap

__author__= "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"
__all__ = ["FISS", "FD", "AlignCube"]

class rawData:
    """
    Read a raw file of the FISS.

    Parameters
    ----------
    file : `str`
        File name of the raw fts data file of the FISS.

    Examples
    --------
    >>> from fisspy.read import rawData
    >>> f = 'D:/fisspy_examples/raw_A.fts'
    >>> raw = rawData(f)
    >>> raw.imshow()
    """
    def __init__(self, file):


        if file.find('A.fts') != -1 or  file.find('B.fts') != -1:
            self.ftype = 'raw'
        scale = 0.16
        self.filename = file
        self.xDelt = scale
        self.yDelt = scale

        self.header = fits.getheader(file)
        self.data = fits.getdata(file)
        self.data = self.data.transpose([1, 0, 2])
        self.ndim = self.header['naxis']
        self.cam = file.split('.fts')[0][-1]
        if self.cam == 'A':
            self.wvDelt = 0.019
        elif self.cam == 'B':
            self.wvDelt = -0.026
        self.nwv = self.header['naxis1']
        self.ny = self.header['naxis2']
        self.nx = self.header['naxis3']
        self.date = self.header['date']
        try:
            self.band = self.header['wavelen'][:4]
        except:
            self.band = str(self.header['gratwvln'])[:4]
        #simple wavelength calibration
        self.wave = (np.arange(self.nwv)-self.nwv//2)*self.wvDelt
        self.centralWavelength = 0.
        self.extentRaster = [0, self.nx*self.xDelt,
                             0, self.ny*self.yDelt]
        self.extentSpectro = [self.wave.min()-self.wvDelt/2,
                              self.wave.max()+self.wvDelt/2,
                              0, self.ny*self.yDelt]

        if self.band == '6562' or self.band =='8542':
            self.set = '1'
        elif self.band == '5889' or self.band == '5434':
            self.set = '2'
        self.cmap = plt.cm.gray

    def getRaster(self, wv, hw=0.05):
        """
        Make a raster image for a given wavelength with in width 2*hw

        Parameters
        ----------
        wv : float
            Referenced wavelength.
        hw : float
            A half-width of wavelength to be integrated
            Default is 0.05

        Example
        -------
        >>> raster = raw.getRaster(0.5)

        """
        self.wv = wv
        return getRaster(self.data, self.wave, wv, self.wvDelt, hw=hw)

    def imshow(self, x=None, y=None, wv=None, scale='minMax',
               sigFactor=3, helpBox=True, **kwargs):
        """
        Draw the interactive image for single band FISS raw data.

        Parameters
        ----------
        x : `float`
            X position that you draw a spectral profile.
            Default is image center.
        y : `float`
            Y position that you draw a spectral profile.
            Default is image center.
        wv : `float`
            Wavelength positin that you draw a raster images.
            Default is central wavelength.
        scale : `string`
            Scale method of colarbar limit.
            Default is minMax.
            option: 'minMax', 'std', 'log'
        sigFactor : `float`
            Factor of standard deviation.
            This is worked if scale is set to be 'std'
        helpBox : `bool`
            Show the interacitve key and simple explanation.
            Default is True

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.pyplot` properties
        """
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass

        if not x:
            X = self.nx//2*self.xDelt
        else:
            X = x
        if not y:
            Y = self.ny//2*self.yDelt
        else:
            Y = y
        if not wv:
            WV = self.centralWavelength
        else:
            WV = wv
        self.x = X
        self.y = Y
        self.wv = wv
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        kwargs['interpolation'] = self.imInterp
        self.iIm = singleBand(self, X, Y, WV,
                                  scale=scale, sigFactor=sigFactor,
                                  helpBox=helpBox, **kwargs)  # Basic resource to make interactive image is `~fisspy.image.tdmap.TDmap`
        plt.show()

    def chRasterClim(self, cmin, cmax):
        self.iIm.chRasterClim(cmin, cmax)

    def chSpectroClim(self, cmin, cmax):
        self.iIm.chSpectroClim(cmin, cmax)

    def chcmap(self, cmap):
        self.iIm.chcmap(cmap)

    def chRaster(self, wv):
        self.iIm.wv = wv
        self.iIm._chRaster()

    def chSpect(self, x, y):
        self.iIm.x = x
        self.iIm.y = y
        self.iIm._chSpect()

class FISS:
    """
    Read a FISS data file (proc or comp).

    Parameters
    ----------
    file: `str`
        File name of the FISS fts data.
    x1: `int`, optional
        A left limit index of the frame along the scan direction
    x2: `int`, optional
        A right limit index of the frame along the scan direction
        If None, read all data from x1 to the end of the scan direction.
    y1: `int`, optional
        A left limit index of the frame along the scan direction
    y2: `int`, optional
        A right limit index of the frame along the scan direction
        If None, read all data from x1 to the end of the scan direction.
    noceff: `int`, optional
        The number of coefficients to be used for
        the construction of frame in a pca file.
    smoothingMethod: `str`, optional
        If it is not given, do not apply the noise filter.
        If 'savgol', apply the Savitzky-Golay noise filter in the wavelength axis.
        If 'gauss', apply the Gaussian noise filter in the wavelength axis.
        Default is None.
    wvCalibMethod: `str`, optional
        Method to calibrate wavelength.
        'simple': calibration with the information of the header.
        'center': calibration with the center of the main line.
        'photo': calibration with the photospheric line and the main line.
        Default is 'simple'.

    Other Parameters
    ----------------
    **kwargs : `~scipy.signal.savgol_filter` properties or `~scipy.ndimage.gaussian_filter1d` properties.

    See also
    --------
    `~scipy.signal.savgol_filter`.
    `~scipy.ndimage.gaussian_filter1d`
    """

    def __init__(self, file, x1=0, x2=None, y1=0, y2=None, ncoeff=False, smoothingMethod=None, wvCalibMethod='simple', **kwargs):
        if file.find('1.fts') != -1:
            self.ftype = 'proc'
        elif file.find('c.fts') != -1:
            self.ftype = 'comp'

        if self.ftype != 'proc' and self.ftype != 'comp':
            raise ValueError("Input file is neither proc nor comp data")

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.filename = file
        self.xDelt = 0.16
        self.yDelt = 0.16

        self.header = getHeader(file)
        self.pfile = self.header.pop('pfile', False)
        self.data = readFrame(file, self.pfile, x1=x1, x2=x2, y1=y1, y2=y2, ncoeff=ncoeff)
        self.ndim = self.header['naxis']
        self.ny, self.nx, self.nwv = self.data.shape
        self.wvDelt = self.header['cdelt1']
        self.dx = self.xDelt
        self.dy = self.yDelt
        self.dwv = self.wvDelt
        self.date = self.header['date']
        try:
            wvln = self.header['wavelen']
            if type(wvln) == str:
                self.band = wvln[:4]
            else:
                wvln = str(wvln)
        except:
            self.band = str(self.header['gratwvln'])[:4]

        self.refProfile = self.data.mean((0,1))
        self.wave = self.wvCalib(method=wvCalibMethod)
        self.cwv = self.centralWavelength = cwv = self.header['crval1']

        self.smoothing = False
        if smoothingMethod is not None:
            self.smoothing = True
        if self.smoothing:
            self.smoothingProf(method=smoothingMethod, **kwargs)
        

        self.line = lineName(cwv)
        if self.line == 'Ha':
            self.cam = 'A'
            self.set = '1'
            self.cmap = cm.ha
        elif self.line == 'Ca':
            self.cam = 'B'
            self.set = '1'
            self.cmap = cm.ca
        elif self.line == 'Na':
            self.cam = 'A'
            self.set = '2'
            self.cmap = cm.na
        elif self.line == 'Fe':
            self.cam = 'B'
            self.set = '2'
            self.cmap = cm.fe

        self.extentRaster = [-self.xDelt/2, (self.nx-0.5)*self.xDelt,
                             -self.yDelt/2, (self.ny-0.5)*self.yDelt]
        self.extentSpectro = [self.wave.min()-self.wvDelt/2,
                              self.wave.max()+self.wvDelt/2,
                              -self.yDelt/2, (self.ny-0.5)*self.yDelt]
        self.lv = None

    def reload(self, x1=0, x2=None, y1=0, y2=None, ncoeff=False, smoothingMethod=None, **kwargs):
        """
        Reload the FISS data.

        Parameters
        ----------
        x1 : `int`, optional
            A left limit index of the frame along the scan direction
        x2 : `int`, optional
            A right limit index of the frame along the scan direction
            If None, read all data from x1 to the end of the scan direction.
        y1 : `int`, optional
            A left limit index of the frame along the scan direction
        y2 : `int`, optional
            A right limit index of the frame along the scan direction
            If None, read all data from x1 to the end of the scan direction.
        noceff : `int`, optional
            he number of coefficients to be used for
            the construction of frame in a pca file.
        smoothingMethod: `str`, optional
            If it is not given, do not apply the noise filter.
            If 'savgol', apply the Savitzky-Golay noise filter in the wavelength axis.
            If 'gauss', apply the Gaussian noise filter in the wavelength axis.
            Default is None.

        Other Parameters
        ----------------
        **kwargs : `~scipy.signal.savgol_filter` properties or `~scipy.ndimage.gaussian_filter1d` properties.

        See also
        --------
        `~scipy.signal.savgol_filter`.
        `~scipy.ndimage.gaussian_filter1d`
        """
        self.data = readFrame(self.filename, self.pfile, x1=x1, x2=x2, y1=y1, y2=y2, ncoeff=ncoeff)
        self.ny, self.nx, self.nwv = self.data.shape
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.extentRaster = [0, self.nx*self.xDelt,
                             0, self.ny*self.yDelt]
        self.extentSpectro = [self.wave.min()-self.wvDelt/2,
                              self.wave.max()+self.wvDelt/2,
                              0, self.ny*self.yDelt]
        
        if smoothingMethod is None:
            self.smoothing = True
        if self.smoothing:
            self.smoothingProf(method=smoothingMethod, **kwargs)

    def wvCalib(self, profile=None, method='photo'):
        """
        Wavelength calibration

        Parameters
        ---------
        profile: `~numpy.ndarray`
            Spectrum
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
        if profile is None:
            pf = self.refProfile
        else:
            pf = profile
        try:
            wv = wvCalib(pf, self.header, method=method)
        except:
            raise ValueError(f"Please change the wvCalibMethod among 'simple', 'center', and 'photo'. Current: {method}")
        return wv

    def corSLA(self, refProf=None, pure=None, eps=0.027, zeta=0.055):
        """
        Correction of spectral line(s) profile for stray linght and far wing red-blue asymmetry.

        Parameters
        ----------
        refProf: `numpy.ndarray`, shape (N,), (optional)
            (Spatially averaged) Reference line profile.
            If None, make refProfile by spatially averaging the Data.
            Default is None.
        pure: `~numpy.ndarry`, (optional)
            True if not blended.
            Please see `~fisspy.correction.get_inform.Pure`
        eps: `float`, (optional)
            Fraction of spatial stray light.
            The default is 0.027
        zeta: `float`, (optional)
            Fration of spectral stray light.
            The default is 0.055

        See Also
        --------
        Chae et al. (2013), https://ui.adsabs.harvard.edu/abs/2013SoPh..288....1C/abstract
        CorStrayLight: correction for stray light.
        CorAsymmetry: correction for far wing red-blue asymmetry.
        """
        if refProf is None:
            ndim = self.data.ndim
            axes = tuple([i for i in range(ndim-1)])
            rp = self.data.mean(axes)
        else:
            rp = refProf

        self.data = corSLA(self.wave, self.data, rp, self.line, pure=pure, eps=eps, zeta=zeta)


    def smoothingProf(self, method='savgol', **kwargs):
        """
        Parameters
        ----------
        method: `str`, optional
            If 'savgol', apply the Savitzky-Golay noise filter in the wavelength axis.
            If 'gauss', apply the Gaussian noise filter in the wavelength axis.
            Default is 'savgol'.

        Other Parameters
        ----------------
        **kwargs : `~scipy.signal.savgol_filter` properties or `~scipy.ndimage.gaussian_filter1d` properties.
        """
        self.data = smoothingProf(self.data, method=method, **kwargs)
        self.smoothing = True
    
    def getRaster(self, wv, hw=0.05):
        """
        Make a raster image for a given wavelength with in width 2*hw

        Parameters
        ----------
        wv : float
            Referenced wavelength.
        hw : float
            A half-width of wavelength to be integrated
            Default is 0.05

        Example
        -------
        >>> from fisspy.read import FISS
        >>> fiss = FISS(file)
        >>> raster = fiss.getRaster(0.5)
        """

        self.wv = wv
        return getRaster(self.data, self.wave, wv, self.wvDelt, hw=hw)



    def lambdameter(self, **kw):
        """
        Calculate the doppler shift by using lambda-meter (bisector) method.

        Parameters
        ----------
        **kw:
            See `~fisspy.analysis.doppler.lambdameter`

        Returns
        -------
        wc : `~numpy.ndarray`
            n dimensional array of central wavelength values.
        intc : `~numpy.ndarray`
            n dimensional array of intensies of the line segment.

        """
        self.hw = kw.get('hw', 0.05)
        self.lwc, self.lic = lambdameter(self.wave, self.data, **kw)
        self.lv = (self.lwc-self.centralWavelength)/self.centralWavelength * ac.c.to('km/s').value
        
    def imshow(self, x=None, y=None, wv=None, scale='log', sigFactor=2, helpBox=True, **kwargs):
        """
        Draw interactive FISS raster, spectrogram and profile for single band.

        Parameters
        ----------
        x : `float`
            X position that you draw a spectral profile.
            Default is image center.
        y : `float`
            Y position that you draw a spectral profile.
            Default is image center.
        wv : `float`
            Wavelength positin that you draw a raster images.
            Default is central wavelength.
        scale : `string`
            Scale method of colarbar limit.
            Default is minMax.
            option: 'minMax', 'std', 'log'
        sigFactor : `float`
            Factor of standard deviation.
            This is worked if scale is set to be 'std'
        helpBox : `bool`
            Show the interacitve key and simple explanation.
            Default is True

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.pyplot` properties
        """
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass

        if x is None:
            X = self.nx//2*self.xDelt
        else:
            X = x
        if y is None:
            Y = self.ny//2*self.yDelt
        else:
            Y = y
        if wv is None:
            WV = self.centralWavelength
        else:
            WV = wv
        self.x = X
        self.y = Y
        self.wv = WV
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        self.cmap = kwargs.pop('cmap', self.cmap)
        kwargs['interpolation'] = self.imInterp
        self.iIm = singleBand(self, X, Y, WV,
                                  scale=scale, sigFactor=sigFactor,
                                  helpBox=helpBox, **kwargs)  # Basic resource to make interactive image is `~fisspy.image.tdmap.TDmap`

    def chRasterClim(self, cmin, cmax):
        self.iIm.chRasterClim(cmin, cmax)

    def chSpectroClim(self, cmin, cmax):
        self.iIm.chSpectroClim(cmin, cmax)

    def chcmap(self, cmap):
        self.iIm.chcmap(cmap)

    def chRaster(self, wv):
        self.iIm.wv = wv
        self.iIm._chRaster()

    def chSpect(self, x, y):
        self.iIm.x = x
        self.iIm.y = y
        self.iIm._chSpect()

    def vshow(self, x=None, y=None, **kw):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass

        if x is None:
            X = self.nx//2*self.xDelt
        else:
            X = x
        if y is None:
            Y = self.ny//2*self.yDelt
        else:
            Y = y

        self.xpix = int(X/self.dx+0.5)
        self.x = self.xpix*self.dx
        self.ypix = int(Y/self.dy+0.5)
        self.y = self.ypix*self.dy
        self.xp0 = self.xpix
        self.yp0 = self.ypix

        self.lambdameter(**kw)
        
        # figure setting
        self.fig = plt.figure(figsize=[18,7])
        gs = gridspec.GridSpec(1, 5)
        self.axI = self.fig.add_subplot(gs[0,0])
        self.axV = self.fig.add_subplot(gs[0,1], sharex=self.axI, sharey=self.axI)
        self.axSpec = self.fig.add_subplot(gs[0,2:])
        self.axI.set_xlabel('X (arcsec)')
        self.axI.set_ylabel('Y (arcsec)')
        self.axSpec.set_xlabel(r'Wavelength ($\AA$)')
        self.axSpec.set_ylabel('Intensity (Count)')
        self.axI.set_title("Intensity")
        self.axV.set_title("Velocity (km/s)")
        self.axSpec.set_title(r"X = %.2f'', Y = %.2f'' (X$_{pix}$ = %i, Y$_{pix}$ = %i), $\Delta\lambda$ = %.2f"%(self.x, self.y, self.xpix, self.ypix, self.hw))
        self.axI.set_xlim(self.extentRaster[0], self.extentRaster[1])
        self.axI.set_ylim(self.extentRaster[2], self.extentRaster[3])
        self.axSpec.set_xlim(self.wave.min(), self.wave.max())
        ym = self.data[self.ypix, self.xpix].min()
        yM = self.data[self.ypix, self.xpix].max()
        margin = (yM-ym)*0.05
        self.axSpec.set_ylim(ym-margin, yM+margin)
        self.axSpec.minorticks_on()
        self.axSpec.tick_params(which='both', direction='in')

        # Draw
        self.imI = self.axI.imshow(self.lic, self.cmap, origin='lower', extent=self.extentRaster)
        tmp = self.lic.copy().flatten()
        tmp.sort()
        m = tmp[200:-200].mean()
        std = tmp[200:-200].std()
        Imin = m-std*3
        Imax = m+std*3
        # Imin = tmp[200:-200].min()
        # Imax = tmp[200:-200].max()
        self.imI.set_clim(Imin, Imax)
        self.imV = self.axV.imshow(self.lv, plt.cm.RdBu_r, origin='lower', extent=self.extentRaster, clim=[-10, 10])
        self.pSpec = self.axSpec.plot(self.wave, self.data[self.ypix, self.xpix], color='k')[0]
        self.pHL = self.axSpec.plot([self.lwc[self.ypix, self.xpix]-self.hw, self.lwc[self.ypix, self.xpix]+self.hw],
                                    [self.lic[self.ypix, self.xpix], self.lic[self.ypix, self.xpix]], color='r')[0]
        self.pVL = self.axSpec.plot([self.lwc[self.ypix, self.xpix], self.lwc[self.ypix, self.xpix]],
                                    [1e4, 0], color='r')[0]
        self.pointI = self.axI.scatter(self.x, self.y, 50, marker='x', color='r')
        self.pointV = self.axV.scatter(self.x, self.y, 50, marker='x', color='r')

        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._onKey_vs)
        self.fig.show()

    def _onKey_vs(self, event):
        if event.key == 'left':
            self.xp0 = self.xpix
            self.yp0 = self.ypix
            if self.xpix > 0:
                self.xpix -= 1
            else:
                self.xpix = self.nx-1
            self.x = self.xpix*self.dx
            self._chPos()
        elif event.key == 'right':
            self.xp0 = self.xpix
            self.yp0 = self.ypix
            if self.xpix < self.nx-1:
                self.xpix += 1
            else:
                self.xpix = 0
            self.x = self.xpix*self.dx
            self._chPos()
        elif event.key == 'down':
            self.xp0 = self.xpix
            if self.ypix > 0:
                self.ypix -= 1
            else:
                self.ypix = self.ny-1
            self.y = self.ypix*self.dy
            self._chPos()
        elif event.key == 'up':
            self.yp0 = self.ypix
            if self.ypix < self.ny-1:
                self.ypix += 1
            else:
                self.ypix = 0
            self.y = self.ypix*self.dy
            self._chPos()
        elif event.key == ' ' and (event.inaxes == self.axI or event.inaxes == self.axV):
            self.xp0 = self.xpix
            self.yp0 = self.ypix
            self.x = event.xdata
            self.y = event.ydata
            self.xpix = int(self.x/self.dx+0.5)
            self.ypix = int(self.y/self.dy+0.5)
            self.x = self.xpix*self.dx
            self.y = self.ypix*self.dy
            self._chPos()
        elif event.key == 'ctrl+b' or event.key == 'cmd+b':
            x = self.xpix
            y = self.ypix
            self.xp = self.xp0
            self.yp = self.yp0
            self.x = self.xpix*self.dx
            self.y = self.ypix*self.dy
            self._chPos()
            self.xp0 = x
            self.yp0 = y


    def _chPos(self):
        self.pSpec.set_ydata(self.data[self.ypix, self.xpix])
        self.pHL.set_xdata([self.lwc[self.ypix, self.xpix]-self.hw, self.lwc[self.ypix, self.xpix]+self.hw])
        self.pHL.set_ydata([self.lic[self.ypix, self.xpix], self.lic[self.ypix, self.xpix]])
        self.pVL.set_xdata([self.lwc[self.ypix, self.xpix], self.lwc[self.ypix, self.xpix]])
        self.pointI.set_offsets([self.x, self.y])
        self.pointV.set_offsets([self.x, self.y])
        self.axSpec.set_ylim(self.data[self.ypix, self.xpix].min()-100,
                                self.data[self.ypix, self.xpix].max()+100)
        self.axSpec.set_title(r"X = %.2f'', Y = %.2f'' (X$_{pix}$ = %i, Y$_{pix}$ = %i), $\Delta\lambda$ = %.2f"%(self.x, self.y, self.xpix, self.ypix, self.hw))
        self.fig.canvas.draw_idle()
    
    def chIclim(self, cmin, cmax):
        self.imI.set_clim(cmin, cmax)
        self.fig.canvas.draw_idle()
    
    def chVclim(self, cmin, cmax):
        self.imV.set_clim(cmin, cmax)
        self.fig.canvas.draw_idle()

class FD:
    """
    Read the FISS Data (FD) file.

    Parameters
    ----------
    fdFile: `str`
        File name of the FISS Data file.
    maskFile: `str`
        File name of the mask file.
    timeFile: `str`
        File name of the time file.
    maskValue: `float`
        Value of the mask pixel.
    spatialAvg: `bool`
        Subtract the spatially averaged value to all pixels.
    timeAvg: `bool`
        Subtract the temporal averaged value to all pixels.
    """
    def __init__(self, fdFile, maskFile, timeFile, maskValue=-1,
                 spatialAvg=False, timeAvg=False):
        self.maskValue = maskValue
        self._spAvg = spatialAvg
        self._timeAvg = timeAvg
        self.ftype = 'FD'
        self.data = fits.getdata(fdFile).astype(float)
        self.fdFile = fdFile
        self.header = fits.getheader(fdFile)
        self.time = fits.getdata(timeFile)
        self.reftpix = np.abs(self.time-0).argmin()
        self.xDelt = self.yDelt = 0.16
        self.min0 = np.min(self.data, axis=(1,2))
        self.max0 = np.max(self.data, axis=(1,2))
        unit = fits.getheader(timeFile)['unit']
        if unit == 'min':
            self.time *= 60

        self.mask = fits.getdata(maskFile).astype(bool)
        self.dt = np.median(self.time-np.roll(self.time, 1))
        self.nt, self.ny, self.nx, self.nid = self.data.shape

        reftime = self.header['reftime']
        self.reftime = _isoRefTime(reftime)
        self.Time = self.reftime + self.time * u.second
        self.timei = self.time-self.time[0]
        self.header['sttime'] = self.Time[0].value
        wid = self.header['ID1'][:2]
        if wid == 'HI':
            self.cmap = [cm.ha]*self.nid

        elif wid == 'Ca':
            self.cmap = [cm.ca]*self.nid
        elif wid == 'Na':
            self.cmap = [cm.na]*self.nid
        elif wid == 'Fe':
            self.cmap = [cm.fe]*self.nid

        try:
            xpos = self.header['xpos']
            ypos = self.header['ypos']
        except:
            xpos = self.header.get('crval1', 0)
            ypos = self.header.get('crval2', 0)
        self.xpos = xpos
        self.ypos = ypos
        xm = xpos - self.nx/2*self.xDelt
        xM = xpos + self.nx/2*self.xDelt
        ym = ypos - self.ny/2*self.yDelt
        yM = ypos + self.ny/2*self.yDelt
        self.extent = [xm, xM, ym, yM]
        self._xar = np.linspace(xm+self.xDelt/2,
                                xM-self.xDelt/2, self.nx)
        self._yar = np.linspace(ym+self.yDelt/2,
                                yM-self.yDelt/2, self.ny)
        if maskValue != -1:
            self._mask(maskValue)
        if spatialAvg:
            self.spatialAverage()
        if timeAvg:
            self.timeAverage()
        self.min = self.min0[self.reftpix]
        self.max = self.max0[self.reftpix]
        self.idh = self.header['ID*']
        for i in range(self.nid):
            if self.idh[i][-1] == 'V':
                self.cmap[i] = plt.cm.RdBu_r
                tmp = np.abs(self.max[i]-self.min[i])/2*0.7
                if tmp > 15:
                    tmp = 0.8
                self.min[i] = -tmp
                self.max[i] = tmp

    def _mask(self, val):
        self.data[np.invert(self.mask),:] = val

    def spatialAverage(self):
        for i in range(self.nt):
            med = np.median(self.data[i,self.mask[i]], 0)
            self.data[i] -= med
            self.min0[i] -= med
            self.max0[i] -= med

    def timeAverage(self):
        med = np.median(self.data, 0)
        self.data -= med
        self.min0 -= np.median(med, (0,1))
        self.max0 -= np.median(med, (0,1))

    def originalData(self, maskValue=-1, spatialAvg=False, timeAvg=False):
        self.data = fits.getdata(self.fdFile).astype(float)
        self.min0 = np.min(self.data, axis=(1,2))
        self.max0 = np.max(self.data, axis=(1,2))
        if maskValue != -1:
            self.maskValue = maskValue
            self._mask(maskValue)
        if spatialAvg:
            self.spatialAverage()
        if timeAvg:
            self.timeAverage()

        self.min = self.min0[self.reftpix]
        self.max = self.max0[self.reftpix]
        for i in range(self.nid):
            if self.idh[i][-1] == 'V':
                self.cmap[i] = plt.cm.RdBu_r
                tmp = np.abs(self.max[i]-self.min[i])/2*0.7
                if tmp > 15:
                    tmp = 0.8
                self.min[i] = -tmp
                self.max[i] = tmp

    def bandpassFilter(self, filterRange):
        for n, i in enumerate(filterRange):
            filterRange[n] = i*1e-3
        self.data = FourierFilter(self.data, self.nt, self.dt, filterRange)
        if self.maskValue != -1:
            self._mask(self.maskValue)
        self.min0 = np.min(self.data, axis=(1,2))
        self.max0 = np.max(self.data, axis=(1,2))
        self.min = self.min0[self.reftpix]
        self.max = self.max0[self.reftpix]
        for i in range(self.nid):
            if self.idh[i][-1] == 'V':
                self.cmap[i] = plt.cm.RdBu_r
                tmp = np.abs(self.max[i]-self.min[i])/2*0.7
                if tmp > 15:
                    tmp = 0.8
                self.min[i] = -tmp
                self.max[i] = tmp

    def imshow(self, x=0, y=0, t=0, cid=0,
               levels=None, maxPeriod=32, helpBox=True, **kwargs):

        self.kwargs = kwargs
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass


        # transpose to pixel position.
        xpix, ypix, tpix = self._pixelPosition(x, y, t)
        self._x0 = self.x
        self._y0 = self.y
        self._t0 = self.t
        self._xh = self.x
        self._yh = self.y
        self._th = self.t
        self.cid = cid
        self._cidh = cid
        self.maxPeriod = maxPeriod


        #Keyboard helpBox
        if helpBox:
            helpFig = plt.figure('Key Help Box', figsize=[3.5, 3])
            ax = helpFig.add_subplot(111)
            ax.set_position([0,0,1,1])
            ax.set_axis_off()
            ax.text(0.05, 0.91, 'ctrl+h: Reset to original setting')
            ax.text(0.05, 0.81, 'ctrl+num: Draw the plot ID = num')
            ax.text(0.05, 0.71, 'ctrl+right: Move to right')
            ax.text(0.05, 0.61, 'ctrl+left: Move to left')
            ax.text(0.05, 0.51, 'ctrl+up: Move to up')
            ax.text(0.05, 0.41, 'ctrl+down: Move to down')
            ax.text(0.05, 0.31, 'right: Next time data')
            ax.text(0.05, 0.21, 'right: Previous time data')
            ax.text(0.05, 0.11, 'spacebar: change to current mouse point')
            ax.text(0.05, 0.01, 'ctrl+b: back to the previous image')


        # Figure setting
        figsize = kwargs.pop('figsize', [10, 8])
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.set_window_title('FISS Data')
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        gs = gridspec.GridSpec(5,5)

        self.axRaster = self.fig.add_subplot(gs[0:3, :2]) # Raster
        self.axRaster.set_xlabel('X (arcsec)')
        self.axRaster.set_ylabel('Y (arcsec)')
        self.axRaster.set_title(self.idh[0])

        self.axTS = self.fig.add_subplot(gs[1:3, 2:]) # TimeSeries
        self.axTS.set_xlabel('Time (sec)')
        self.axTS.set_ylabel('Intensity (count)')
        self.axTS.set_xlim(self.timei[0], self.timei[-1])
        self.axTS.minorticks_on()
        self.axTS.tick_params(which='both', direction='in')
        self.axTS.set_title('Time series')

        self.axWavelet = self.fig.add_subplot(gs[3:, 2:])
        self.axWavelet.set_title('Wavelet Power Spectrum')
        self.axWavelet.set_xlabel('Time (sec)')
        self.axWavelet.set_ylabel('Period (minute)')
        self.axWavelet.set_xlim(self.timei[0], self.timei[-1])
        self.axWavelet.set_yscale('symlog', basey=2)
        self.axWavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axWavelet.ticklabel_format(axis='y',style='plain')
        self.axWavelet.set_ylim(self.maxPeriod, 0.5)

        self.axPower = self.fig.add_subplot(gs[3:, :2])
        self.axPower.set_title('Power Spectrum')

        self.axPower.set_ylabel('Period (minute)')
        self.axPower.set_ylim(self.maxPeriod, 0.5)
        self.axPower.set_yscale('symlog', basey=2)
        self.axPower.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axPower.ticklabel_format(axis='x',style='sci', scilimits=(0,1))
        self.axPower.minorticks_on()
        self.axPower.tick_params(which='both', direction='in')

        # Plot
        data = self.data[:, ypix, xpix, self.cid]
        self.imRaster = self.axRaster.imshow(self.data[tpix,:,:,cid],
                                             self.cmap[cid],
                                             origin='lower',
                                             extent=self.extent,
                                             clim=[self.min[cid],
                                                   self.max[cid]],
                                             interpolation=self.imInterp)
        self.timeseries = self.axTS.plot(self.timei,
                                         data,
                                         color='k')[0]

        #wavelet
        if not levels:
            levels = [0.1, 0.25, 0.4,
                      0.55, 0.7, 1]
        self.levels = levels
        self._plotWavelet(xpix, ypix)
#        divider = make_axes_locatable(self.axWavelet)
#        cax = divider.append_axes('right', size='5%', pad=0.1)
#        plt.colorbar(self.contourIm, cax=cax)

        #gws
        self.powerGWS = self.axPower.plot(self.gws, self.period, color='k',
                                          label='GWS')[0]
        #lws
        self.lws = self.wavelet.power[:, tpix]
        self.powerLWS = self.axPower.plot(self.lws, self.period,
                                          color='r', label='LWS')[0]
        self.axPower.legend()

        # marker
        self.point = self.axRaster.scatter(self.x, self.y, 50,
                                           marker='x',
                                           color='r')
        self.vlineTS = self.axTS.axvline(self.t,
                                         ls='dashed',
                                         color='b')
        self.vlineWavelet = self.axWavelet.axvline(self.t,
                                                   ls='dashed',
                                                   color='k')
        peakPGWS = self.period[self.gws.argmax()]
        peakPLWS = self.period[self.lws.argmax()]
        self.hlineGWS = self.axPower.axhline(peakPGWS,
                                             ls='dotted',
                                             color='k')
        self.hlineLWS = self.axPower.axhline(peakPLWS,
                                             ls='dotted',
                                             color='r')

        #infoBox
        self.axInfo = self.fig.add_subplot(gs[0, 2:])
        self.axInfo.set_axis_off()
        self.isotInfo = self.axInfo.text(0.05, 0.8,
                                    '%s'%self.Time[self.tpix].value,
                                    fontsize=12)
        self.tInfo = self.axInfo.text(0.05, 0.55,
                                 't=%i sec (tpix=%i)'%(self.t, self.tpix),
                                 fontsize=12)
        self.posiInfo = self.axInfo.text(0.05, 0.3,
                        "X=%.1f'', Y=%.1f'' (xpix=%i, ypix=%i)"%(self.x,
                                                                 self.y,
                                                                 xpix,
                                                                 ypix),
                                            fontsize=12)
        self.peakPeriodGWS = self.axInfo.text(0.05, -0.1,
                                         r'P$_{peak, GWS}$=%.2f min'%peakPGWS,
                                         fontsize=12)
        self.peakPeriodLWS = self.axInfo.text(0.05, -0.35,
                                         r'P$_{peak, LWS}$=%.2f min'%peakPLWS,
                                         fontsize=12)

        #Axis limit
        self.axTS.set_ylim(data.min(), data.max())
        self.axPower.set_xlim(0, self.lpmax)
        self.axWavelet.set_aspect(adjustable='box', aspect='auto')
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()

    def _on_key(self, event):
        if event.key == 'ctrl+right':
            if self.x < self._xar[-1]:
                self.x += self.xDelt
            else:
                self.x = self._xar[0]
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
        elif event.key == 'ctrl+left':
            if self.x > self._xar[0]:
                self.x -= self.xDelt
            else:
                self.x = self._xar[-1]
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
        elif event.key == 'ctrl+up':
            if self.y < self._yar[-1]:
                self.y += self.yDelt
            else:
                self.y = self._yar[0]
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
        elif event.key == 'ctrl+down':
            if self.y > self._yar[0]:
                self.y -= self.yDelt
            else:
                self.y = self._yar[-1]
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
        elif event.key == 'right':
            if self.tpix < self.nt-1:
                self.tpix += 1
            else:
                self.tpix = 0
            self.t = self.timei[self.tpix]
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
        elif event.key == 'left':
            if self.tpix > 0:
                self.tpix -= 1
            else:
                self.tpix = self.nt-1
            self.t = self.timei[self.tpix]
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
        elif event.key == ' ' and event.inaxes == self.axRaster:
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
            self.x = event.xdata
            self.y = event.ydata
        elif event.key == ' ' and (event.inaxes == self.axTS or
                                   event.inaxes == self.axWavelet):
            self.t = event.xdata
            self._xb = self._x0
            self._yb = self._y0
            self._tb = self._t0
            self.tpix = np.abs(self.timei-self.t).argmin()
            self.t = self.timei[self.tpix]
        elif event.key == 'ctrl+b':
            x = self.x
            y = self.y
            t = self.t
            self.x = self._xb
            self.y = self._yb
            self.t = self._tb
            self._xb = x
            self._yb = y
            self._tb = t
            self.tpix = np.abs(self.timei-self.t).argmin()
        elif event.key == 'ctrl+h':
            self.x = self._xh
            self.y = self._yh
            self.t = self._th
            self.tpix = np.abs(self.timei-self.t).argmin()
            self.cid = self._cidh
            self._changeID()
            self.axRaster.set_title(self.idh[self.cid])
            self.imRaster.set_cmap(self.cmap[self.cid])
        for iid in range(self.nid):
            if event.key == 'ctrl+%i'%iid:
                self.cid = iid
                self._changeID()
                self.axRaster.set_title(self.idh[iid])
                self.imRaster.set_cmap(self.cmap[self.cid])
                if self.idh[iid][-1] == 'V':
                    self.axTS.set_ylabel('Velocity (km/s)')
                else:
                    self.axTS.set_ylabel('Intensity (Count)')

        if self.x != self._x0 or self.y != self._y0:
            xpix, ypix, tpix = self._pixelPosition(self.x, self.y,
                                                   self.t)
            self._changeWavelet(xpix, ypix)
            self._changePlot(xpix, ypix)
            self._x0 = self.x
            self._y0 = self.y
            self.posiInfo.set_text(
                    "X=%.1f'', Y=%.1f'' (xpix=%i, ypix=%i)"%(self.x,
                                                             self.y,
                                                             xpix,
                                                             ypix))
        if self.t != self._t0:
            self._changeRaster()
            self.lws = self.wavelet.power[:, self.tpix]
            self.powerLWS.set_xdata(self.lws)
            self.vlineTS.set_xdata(self.t)
            self.vlineWavelet.set_xdata(self.t)
            peakPLWS = self.period[self.lws.argmax()]
            self.hlineLWS.set_ydata(peakPLWS)
            self._t0 = self.t
            self.isotInfo.set_text('%s'%self.Time[self.tpix].value)
            self.tInfo.set_text('t=%i sec (tpix=%i)'%(self.t, self.tpix))
            self.peakPeriodLWS.set_text(
                    r'P$_{peak, LWS}$=%.2f min'%peakPLWS)
        self.fig.canvas.draw_idle()

    def _changeID(self):
        xpix, ypix, tpix = self._pixelPosition(self.x, self.y,
                                                   self.t)
        self._changeWavelet(xpix, ypix)
        self._changePlot(xpix, ypix)
        self._changeRaster()
        self.imRaster.set_clim(self.min[self.cid],
                               self.max[self.cid])

    def _changePlot(self, xpix, ypix):
        data = self.data[:, ypix, xpix, self.cid]
        self.timeseries.set_ydata(data)
        self.axTS.set_ylim(data.min(), data.max())
        self.powerGWS.set_xdata(self.gws)
        self.lws = self.wavelet.power[:, self.tpix]
        self.powerLWS.set_xdata(self.lws)
        self.point.set_offsets([self.x, self.y])
        peakPGWS = self.period[self.gws.argmax()]
        peakPLWS = self.period[self.lws.argmax()]
        self.hlineGWS.set_ydata(peakPGWS)
        self.hlineLWS.set_ydata(peakPLWS)
        self.peakPeriodGWS.set_text(
                    r'P$_{peak, GWS}$=%.2f min'%peakPGWS)
        self.peakPeriodLWS.set_text(
                    r'$P_{peak, LWS}$=%.2f min'%peakPLWS)
        self.axPower.set_xlim(0, self.lpmax)

    def _changeRaster(self):
        self.imRaster.set_data(self.data[self.tpix, :, :, self.cid])

    def _pixelPosition(self, x, y, t):
        tpix = np.abs(self.timei-t).argmin()
        xpix = np.abs(self._xar-x).argmin()
        ypix = np.abs(self._yar-y).argmin()
        self.x = self._xar[xpix]
        self.y = self._yar[ypix]
        self.t = self.timei[tpix]
        self.tpix = tpix
        return xpix, ypix, tpix

    def _changeWavelet(self, xpix, ypix):
        self.axWavelet.cla()
        self._plotWavelet(xpix, ypix)

    def _plotWavelet(self, xpix, ypix):
        self.wavelet = Wavelet(self.data[:, ypix, xpix, self.cid],
                       self.dt, **self.kwargs)
        self.lpmax = self.wavelet.power.max()
        self.period = self.wavelet.period/60
        self.gws = self.wavelet.gws
        wpower = self.wavelet.power/self.wavelet.power.max()
        self.contour = self.axWavelet.contourf(self.timei, self.period,
                                               wpower, len(self.levels),
                                               colors=['w'])
        self.contourIm = self.axWavelet.contourf(self.contour,
                                                 levels=self.levels
                                                 )
        self.axWavelet.fill_between(self.timei, self.wavelet.coi/60,
                                    self.period.max(), color='grey',
                                    alpha=0.4, hatch='x')
        self.axWavelet.set_title('Wavelet Power Spectrum')
        self.axWavelet.set_xlabel('Time (sec)')
        self.axWavelet.set_ylabel('Period (minute)')
        self.axWavelet.set_xlim(self.timei[0], self.timei[-1])
        self.axWavelet.set_yscale('symlog', basey=2)
        self.axWavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axWavelet.ticklabel_format(axis='y',style='plain')
        self.vlineWavelet = self.axWavelet.axvline(self.t,
                                                   ls='dashed',
                                                   color='k')
        self.axWavelet.set_ylim(self.maxPeriod, 0.5)

    def chLevels(self, levels):
        """
        """
        self.levels = levels
        xpix, ypix, tpix = self._pixelPosition(self.x, self.y, self.t)
        self._changeWavelet(xpix, ypix)

    def chInterp(self, interp):
        """
        """
        self.imInterp = interp
        self.imRaster.set_interpolation(interp)

    def chBPFilter(self, filterRange):
        """
        """
        self.originalData(maskValue=self.maskValue, spatialAvg=self._spAvg,
                          timeAvg=self._timeAvg)
        self.bandpassFilter(filterRange)

    def chRasterClim(self, cmin, cmax):
        """
        """
        self.imRaster.set_clim(cmin, cmax)

    def chPosition(self, x, y):
        """
        """
        self.x = x
        self.y = y
        self._x0 = x
        self._y0 =y
        xpix, ypix, tpix = self._pixelPosition(x, y, self.t)
        self._changeWavelet(xpix, ypix)
        self._changePlot(xpix, ypix)
        self.posiInfo.set_text(
                    "X=%.1f'', Y=%.1f'' (xpix=%i, ypix=%i)"%(self.x,
                                                             self.y,
                                                             xpix,
                                                             ypix))

    def chtime(self, t):
        """
        """
        self.t = t
        self._t0 = t
        self._changeRaster()
        self.lws = self.wavelet.power[:, self.tpix]
        self.LWS.set_xdata(self.lws)
        self.vlineTS.set_xdata(self.t)
        self.vlineWavelet.set_xdata(self.t)
        peakPLWS = self.period[self.lws.argmax()]
        self.hlineLWS.set_ydata(peakPLWS)
        self._t0 = self.t
        self.isotInfo.set_text('%s'%self.Time[self.tpix].value)
        self.tInfo.set_text('t=%i sec (tpix=%i)'%(self.t, self.tpix))
        self.peakPeriodLWS.set_text(
                r'P$_{peak, LWS}$=%.2f min'%peakPLWS)

    # def TD(self, ID=0, filterRange=None):
    #     hdu = fits.PrimaryHDU(self.data[:,:,:,ID])
    #     h= hdu.header
    #     h['cdelt1'] = self.xDelt
    #     h['cdelt2'] = self.yDelt
    #     h['cdelt3'] = self.dt
    #     h['crval1'] = self.xpos
    #     h['crval2'] = self.ypos
    #     h['sttime'] = self.Time[0].value

    #     return TDmap(self.data[:,:,:,ID], h, self.time,
    #                  filterRange=filterRange, cmap=self.cmap[ID])

    def set_clim(self, cmin, cmax):
        self.imRaster.set_clim(cmin, cmax)

class calibData:
    """
    Read the calibration file such as 'BiasDark', 'Flat', 'FLAT' and 'SLIT'.

    Parameters
    ----------
    file : str

    """

    def __init__(self, file):

        if file.find('BiasDark') != -1:
            self.ftype = 'BiasDark'
        elif file.find('Flat') != -1:
            self.ftype = 'Flat'
        elif file.find('FLAT') != -1:
            self.ftype = 'FLAT'
        elif file.find('SLIT') != -1:
            self.ftype = 'SLIT'

        self.data = fits.getdata(file)
        self.header = fits.getheader(file)

        self.nx = self.header['naxis1']
        self.ny = self.header['naxis2']
        if self.ftype == 'Flat':
            self.nf = self.header['naxis3']

        if file.find('_A') != -1:
            self.cam = 'A'
        elif file.find('_B') != -1:
            self.cam = 'B'

    def imshow(self):
        """
        """
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass

        self.fig, self.ax = plt.subplots(figsize=[10, 6])
        if self.ftype != 'Flat':
            self.image = self.ax.imshow(self.data, origin='lower',
                                        cmap = plt.cm.gray)
            self.fig.tight_layout()
        else:
            self.num = 0
            self.num0 = self.num
            self.image = self.ax.imshow(self.data[self.num], origin='lower',
                                        cmap = plt.cm.gray)
            self.fig.tight_layout()
            self.fig.canvas.mpl_connect('key_press_event', self._onKey)

    def _onKey(self, event):
        if event.key == 'right':
            if self.num < self.nf-1:
                self.num += 1
            else:
                self.num = 0
        elif event.key == 'left':
            if self.num > 0:
                self.num -= 1
            else:
                self.num = self.nf-1
        if self.num != self.num0:
            self.image.set_data(self.data[self.num])
            self.num0 = self.num
        self.fig.canvas.draw_idle()

class AlignCube:
    """
    Read align cube.

    Parameters
    ----------
    fname: `str`
        File name of the align data cube.
        
    Returns
    -------
    None
    """
    def __init__(self, fname):
        res = np.load(fname)
        self.data = res['data']
        self.time = res['time']
        self.dt = res['dt']
        self.dx = res['dx']
        self.dy = res['dy']

    def imshow(self, **kwargs):
        """
        Show align cube and make Time-Distance map.

        Other Parameters
        ----------------
        **kwargs: `.makeTDmap` properties (optional)
            Keyword arguments
            
        Returns
        -------
        None
        """
        self.td = makeTDmap(self.data, dx=self.dx, dy=self.dy, dt=self.dt, **kwargs)

def _isoRefTime(refTime):
    year = refTime[:4]
    month = refTime[4:6]
    day = refTime[6:8]
    hour = refTime[9:11]
    minute = refTime[11:13]
    sec = refTime[13:15]
    isot = '%s-%s-%sT%s:%s:%s'%(year, month, day, hour, minute, sec)
    return Time(isot)
