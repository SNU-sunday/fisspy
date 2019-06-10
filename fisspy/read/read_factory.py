from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve as conv
from fisspy import cm
import matplotlib.pyplot as plt
from astropy.constants import c
from fisspy.analysis.doppler import lambdameter
from fisspy.image import interactive_image as IAI
from fisspy.read.readbase import getRaster, getHeader, readFrame
from fisspy.analysis.filter import FourierFilter
from astropy.time import Time
import astropy.units as u
from scipy.fftpack import fft, fftfreq
from matplotlib import gridspec
from fisspy.analysis.wavelet import Wavelet
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ["rawData", "FISS", "FD"]

class rawData:
    """
    Read a raw data of the FISS.
    
    Parameters
    ----------
    file : `str`
        File name of the raw fts data file of the FISS.
    
    Examples
    --------
    
    """
    
    def __init__(self, file, scale=0.16):
        if file.find('A.fts') != -1 or  file.find('B.fts') != -1:
            self.ftype = 'raw'
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
        self.band = self.header['wavelen'][:4]
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
        >>> from fisspy.read import rawData
        >>> raw = rawData(file)
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
        if not x:
            x = self.nx//2*self.xDelt
        if not y:
            y = self.ny//2*self.yDelt
        if not wv:
            wv = self.centralWavelength
        self.x = x
        self.y = y
        self.wv = wv
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        kwargs['interpolation'] = self.imInterp
        self.iIm = IAI.singleBand(self, x, y, wv,
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
    
class FISS(object):
    """
    Read a FISS data file (proc or comp).
    
    Parameters
    ----------
    file : `str`
        File name of the FISS fts data.
    noiseSuprresion : `bool`, optional
        If True Savitzky-Golay noise filter is applied in the wavelength axis.
        Default is False.
    simpleWvCalib : `bool`, optional
        If True wavelength is simply calibrated by using the header parameters.
        Default is True.
    absScale : `bool`, optional
        If False the central wavelength is set to be zero.
        If True the central wavelength is set to be wavelength at lab frame.
        It works if simpleWvCalibration is True.
        Default is True
    
    Other Parameters
    ----------------
    **kwargs : `~scipy.signal.svagol_filter` properties
    
    See also
    --------
    `~scipy.signal.savgol_filter`
    
    Examples
    --------
    >>> from fisspy import read
    >>> import fisspy.data.sample
    >>> fiss = read.FISS(fisspy.data.sample.FISS_IMAGE)
    """
    
    def __init__(self, file, ncoeff=False, noiseSuppression=False,
                 simpleWaveCalib=True, absScale=True, **kwargs):
        if file.find('1.fts') != -1:
            self.ftype = 'proc'
        elif file.find('c.fts') != -1:
            self.ftype = 'comp'
        
        if self.ftype != 'proc' and self.ftype != 'comp':
            raise ValueError("Input file is neither proc nor comp data")
            
        self.filename = file
        self.xDelt = 0.16
        self.yDelt = 0.16
        
        self.header = getHeader(file)
        self.pfile = self.header.pop('pfile',False)
        self.data = readFrame(file, self.pfile, ncoeff=ncoeff)
        self.ndim = self.header['naxis']
        self.nwv = self.header['naxis1']
        self.ny = self.header['naxis2']
        self.nx = self.header['naxis3']
        self.wvDelt = self.header['cdelt1']
        self.date = self.header['date']
        self.band = self.header['wavelen'][:4]
        
        self.refProfile = self.data.mean((0,1))
        self.wave = self._waveCalibration(simpleWaveCalib= simpleWaveCalib,
                                        absScale= absScale, **kwargs)
        
        self.noiseSuppression = noiseSuppression
        if noiseSuppression:
            self._noiseSuppression()
        
        if self.band == '6562':
            self.cam = 'A'
            self.set = '1'
            self.cmap = cm.ha
        elif self.band == '8542':
            self.cam = 'B'
            self.set = '1'
            self.cmap = cm.ca
        elif self.band == '5889':
            self.cam = 'A'
            self.set = '2'
            self.cmap = cm.na
        elif self.band == '5434':
            self.cam = 'B'
            self.set = '2'
            self.cmap = cm.fe
            
        self.extentRaster = [0, self.nx*self.xDelt,
                             0, self.ny*self.yDelt]
        self.extentSpectro = [self.wave.min()-self.wvDelt/2,
                              self.wave.max()+self.wvDelt/2,
                              0, self.ny*self.yDelt]
            
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

    
    def _waveCalibration(self, simpleWaveCalib= True, absScale= True,
                         **kwargs):
        """
        Wavelength calibration
        
        If SimpleWvCalib is True, the wavelength is calibrated by using information in header.
        If absScale is True, the central wavelength is set to be wavelength in the lab frame, 
        but if absScale is False, the central wavelength is set to be zero.
        """
        method = kwargs.pop('method', True)
        if simpleWaveCalib:
            if absScale:
                self.centralWavelength = self.header['crval1']
                return (np.arange(self.nwv) -
                        self.header['crpix1']) * self.header['cdelt1'] + self.header['crval1']
            else:
                self.centralWavelength = 0
                return (np.arange(self.nwv) -
                        self.header['crpix1']) * self.header['cdelt1']
        else:
            if method:
                if self.band == '6562':
                    line=np.array([6561.097,6564.206])
                    lamb0=6562.817
                    dldw=0.019182
                elif self.band == '8542':
                    line=np.array([8540.817,8546.222])
                    lamb0=8542.090
                    dldw=-0.026252
                elif self.band == '5889':
                    line=np.array([5889.951,5892.898])
                    lamb0=5889.9509
                    dldw=0.016847
                elif self.band == '5434':
                    line=np.array([5434.524,5436.596])
                    lamb0=5434.5235
                    dldw=-0.016847
            else:
                if self.band == '6562':
                    line=np.array([6562.817,6559.580])
                    lamb0=6562.817
                    dldw=0.019182
                elif self.band == '8542':
                    line=np.array([8542.089,8537.930])
                    lamb0=8542.090
                    dldw=-0.026252
        
        w = np.arange(self.nwv)
        wl = np.zeros(2)
        wc = self.refProfile[20:self.nwv-20].argmin() + 20
        lamb = (w - wc) * dldw + lamb0
        
        for i in range(2):
            mask = np.abs(lamb - line[i]) <= 0.3
            wtmp = w[mask]
            ptmp = conv(self.refProfile[mask], [-1, 2, -1], 'same')
            mask2 = ptmp[1:-1].argmin() + 1
            try:
                wtmp = wtmp[mask2-3:mask2+4]
                ptmp = ptmp[mask2-3:mask2+4]
            except:
                raise ValueError('Fail to wavelength calibration\n'
                'please change the method %s to %s' %(repr(method), repr(not method)))
            c = np.polyfit(wtmp - np.median(wtmp), ptmp, 2)
            wl[i] = np.median(wtmp) - c[1]/(2*c[0])
            
        dldw = (line[1] - line[0])/(wl[1] - wl[0])
        wc = wl[0] - (line[0] - lamb0)/dldw
        return (w - wc) * dldw
    
    def _noiseSuppression(self, **kwargs):
        window_length = kwargs.pop('window_length', 7)
        polyorder = kwargs.pop('polyorder', 2)
        deriv = kwargs.pop('deriv', 0)
        delta = kwargs.pop('delta', 1.0)
        mode = kwargs.pop('mode', 'interp')
        cval = kwargs.pop('cval', 0.0)
        
        self.data = savgol_filter(self.data, window_length, polyorder,
                                   deriv= deriv, delta= delta, cval= cval,
                                   mode= mode)
        self.noiseSuppression = True
    
        
    def lambdaMeter(self, hw= 0.03, sp= 5e3, wvRange= False,
                    wvinput= True, shift2velocity= True):
        """
        """
        lineShift, intensity = lambdameter(self.wave, self.data,
                                           ref_spectrum= self.refProfile,
                                           wvRange= wvRange, hw= hw,
                                           wvinput= wvinput)
        
        if shift2velocity:
            LOSvelocity = lineShift * c.to('km/s').value/self.centralWavelength
            return LOSvelocity, intensity
        else:
            return lineShift, intensity
        
    def imshow(self, x=None, y=None, wv=None, scale='minMax',
               sigFactor=3, helpBox=True, **kwargs):
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
        
        if not x:
            x = self.nx//2*self.xDelt
        if not y:
            y = self.ny//2*self.yDelt
        if not wv:
            wv = self.centralWavelength
        self.x = x
        self.y = y
        self.wv = wv
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        self.cmap = kwargs.pop('cmap', self.cmap)
        kwargs['interpolation'] = self.imInterp
        self.iIm = IAI.singleBand(self, x, y, wv,
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
        
class FD:
    """
    Read the FISS DATA (FD) file.
    """
    def __init__(self, fdFile, maskFile, timeFile, maskValue=-1,
                 spatialAvg=False, timeAvg=False):
        self.maskValue = maskValue
        self.ftype = 'FD'
        self.data = fits.getdata(fdFile).astype(float)
        self.odata = self.data.copy()
        self.header = fits.getheader(fdFile)
        self.time = fits.getdata(timeFile)
        self.reftpix = np.abs(self.time-0).argmin()
        self.xDelt = self.yDelt = 0.16
        self.min = np.min(self.data, axis=(1,2))
        self.max = np.max(self.data, axis=(1,2))
        unit = fits.getheader(timeFile)['unit']
        if unit == 'min':
            self.time *= 60
        
        self.mask = fits.getdata(maskFile).astype(bool)
        self.dt = np.median(self.time-np.roll(self.time, 1))
        self.nt, self.ny, self.nx, self.nid = self.data.shape
        
        reftime = self.header['reftime']
        self.reftime = _isoRefTime(reftime)
        self.isotime = self.reftime + self.time * u.second
        self.timei = self.time-self.time[0]
        
        wid = self.header['ID1'][:2]
        if wid == 'HI':
            self.cmap = [cm.ha]*self.nid
            
        elif wid == 'Ca':
            self.cmap = [cm.ca]*self.nid
        elif wid == 'Na':
            self.cmap = [cm.na]*self.nid
        elif wid == 'Fe':
            self.cmap = [cm.fe]*self.nid
        
        xpos = self.header.get('xpos', False)
        if xpos:
            ypos = self.header['ypos']
            xm = xpos - self.nx/2*self.xDelt
            xM = xpos + self.nx/2*self.xDelt
            ym = ypos - self.ny/2*self.yDelt
            yM = ypos + self.ny/2*self.yDelt
        else:
            xm = -self.nx/2*self.xDelt
            xM = self.nx/2*self.xDelt
            ym = -self.ny/2*self.yDelt
            yM = self.ny/2*self.yDelt
        self.extent = [xm, xM, ym, yM]
        self._xar = np.linspace(xm+self.xDelt/2,
                                xM-self.xDelt/2, self.nx)
        self._yar = np.linspace(ym+self.yDelt/2,
                                yM-self.yDelt/2, self.ny)
        if maskValue != -1:
            self._mask(maskValue)
        if spatialAvg:
            self._spatialAverage()
        if timeAvg:
            self._timeAverage()
        self._PowerSpectrum()
        self.min = self.min[self.reftpix]
        self.max = self.max[self.reftpix]
        self.idh = self.header['ID*']
        for i in range(self.nid):
            if self.idh[i][-1] == 'V':
                self.cmap[i] = plt.cm.RdBu_r
                tmp = np.abs(self.max[i]-self.min[i])/2*0.7
                if tmp > 15:
                    tmp = 0.8
                self.min[i] = -tmp
                self.max[i] = tmp
            
    def _PowerSpectrum(self):
        self.freq = (fftfreq(self.nt, self.dt)*1e3)[:self.nt//2]
        
        self.power = (np.abs(fft(self.data, axis=0))**2)[:self.nt//2]
        
    def _mask(self, val):
        self.data[np.invert(self.mask),:] = val
        
    def _spatialAverage(self):
        for i in range(self.nt):
            med = np.median(self.data[i,self.mask[i]], 0)
            self.data[i] -= med
            self.min[i] -= med
            self.max[i] -= med
            
    def _timeAverage(self):
        med = np.median(self.data, 0)
        self.data -= med
        self.min -= np.median(med, (0,1))
        self.max -= np.median(med, (0,1))
        
    def originalData(self, maskValue=-1, spatialAvg=False, timeAvg=False):
        self.data = self.odata
        if maskValue != -1:
            self.maskValue = maskValue
            self._mask(maskValue)
        if spatialAvg:
            self._spatialAverage()
        if timeAvg:
            self._timeAverage()
            
    def bandpassFilter(self, filterRange):
        self.data = FourierFilter(self.data, self.nt, self.dt, filterRange)
        if self.maskValue != -1:
            self._mask(self.maskValue)
            
    def imshow(self, x=0, y=0, t=0, cid=0, scale='minMax', **kwargs):
        
        self.kwargs = kwargs
        
        #Scale setting
        if scale == 'log':
            for i in range(self.nid):
                if self.idh[i][-1] != 'V':
                    self.min = np.log(self.min)
                    self.max = np.log(self.max)
        elif scale != 'minMax':
            raise ValueError("scale must be either 'minMax' or 'log'.")
        
        # transpose to pixel position.
        xpix, ypix, tpix = self._transposedPosition(x, y, t)
        self.x0 = self.x
        self.y0 = self.y
        self.t0 = self.t
        self.xh = self.x
        self.yh = self.y
        self.th = self.t
        self.cid = cid
        
        # Figure setting
        figsize = kwargs.pop('figsize', [10, 8])
        self.fig = plt.figure('FISS DATA', figsize=figsize)
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        gs = gridspec.GridSpec(3,3)
        self.axRaster = self.fig.add_subplot(gs[:, 0])
        self.axTS = self.fig.add_subplot(gs[0, 1:]) # TimeSeries
        self.axPower = self.fig.add_subplot(gs[1, 1:])
        self.axWavelet = self.fig.add_subplot(gs[2, 1:])
        self.axRaster.set_xlabel('X (arcsec)')
        self.axRaster.set_ylabel('Y (arcsec)')
        self.axTS.set_xlabel('Time (sec)')
        self.axTS.set_ylabel('Intensity (count)')
        self.axPower.set_xlabel('Frequency (mHz)')
        self.axPower.set_ylabel('Power')
        self.axPower.set_xlim(0, 10)
        self.axTS.set_xlim(self.timei[0], self.timei[-1])
        self.axTS.minorticks_on()
        self.axPower.minorticks_on()
        self.axTS.tick_params(which='both', direction='in')
        self.axPower.tick_params(which='both', direction='in')
        self.axRaster.set_title(self.idh[0])
        self.axTS.set_title('Time series')
        self.axPower.set_title('Fourier Power Spectrum')
        self.axWavelet.set_title('Wavelet Power Spectrum')
        self.axWavelet.set_xlabel('Time (sec)')
        self.axWavelet.set_ylabel('Period (minute)')
        self.axWavelet.set_xlim(self.timei[0], self.timei[-1])
        self.axWavelet.set_yscale('log', basey=2)
        self.axWavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axWavelet.ticklabel_format(axis='y',style='plain')
        self.axWavelet.set_ylim(32, 1)
        
        # Plot
        data = self.data[:, ypix, xpix, self.cid]
        power = self.power[:, ypix, xpix, self.cid]
        power /= power.max()
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
        self.powerSpectrum = self.axPower.plot(self.freq,
                                               power,
                                               color='k')[0]
        #wavelet 
        levels = [0.1, 0.25, 0.4, 
                  0.55, 0.7, 1]
        self.levels = levels
        self._plotWavelet(xpix, ypix)
        divider = make_axes_locatable(self.axWavelet)
        cax = divider.append_axes('right', size='5%', pad=0)
        plt.colorbar(self.contourIm, cax=cax)
        
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
        
        #Axis limit
        self.axTS.set_ylim(data.min(), data.max())
        self.axPower.set_ylim(0, 1)
        self.axWavelet.set_aspect(adjustable='box', aspect='auto')
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _on_key(self, event):
        if event.key == 'ctrl+right':
            if self.x < self._xar[-1]:
                self.x += self.xDelt
            else:
                self.x = self._xar[0]
        elif event.key == 'ctrl+left':
            if self.x > self._xar[0]:
                self.x -= self.xDelt
            else:
                self.x = self._xar[-1]
        elif event.key == 'ctrl+up':
            if self.y < self._yar[-1]:
                self.y += self.yDelt
            else:
                self.y = self._yar[0]
        elif event.key == 'ctrl+down':
            if self.y > self._yar[0]:
                self.y -= self.yDelt
            else:
                self.y = self._yar[-1]
        elif event.key == 'right':
            if self.tpix < self.nt-1:
                self.tpix += 1
            else:
                self.tpix = 0
            self.t = self.timei[self.tpix]
        elif event.key == 'left':
            if self.tpix > 0:
                self.tpix -= 1
            else:
                self.tpix = self.nt-1
            self.t = self.timei[self.tpix]
        elif event.key == 'ctrl+ ' and event.inaxes == self.axRaster:
            self.x = event.xdata
            self.y = event.ydata
        elif event.key == 'ctrl+ ' and (event.inaxes == self.axTS or
                                        event.inaxes == self.axWavelet):
            self.t = event.xdata
            self.tpix = np.abs(self.timei-self.t).argmin()
            self.t = self.timei[self.tpix]
        for iid in range(self.nid):
            if event.key == 'ctrl+%i'%iid:
                self.cid = iid
                self._changeID()
                self.imRaster.set_cmap(self.cmap[self.cid])
                
        if self.x != self.x0 or self.y != self.y0:
            xpix, ypix, tpix = self._transposedPosition(self.x, self.y,
                                                   self.t)
            self._changePlot(xpix, ypix)
            self._changeWavelet(xpix, ypix)
            self.x0 = self.x
            self.y0 = self.y
        elif self.t != self.t0:
            self._changeRaster()
            self.vlineTS.set_xdata(self.t)
            self.vlineWavelet.set_xdata(self.t)
            self.t0 = self.t
        self.fig.canvas.draw_idle()
        
    def _changeID(self):
        xpix, ypix, tpix = self._transposedPosition(self.x, self.y,
                                                   self.t)
        self._changePlot(xpix, ypix)
        self._changeWavelet(xpix, ypix)
        self._changeRaster()
        self.imRaster.set_clim(self.min[self.cid],
                               self.max[self.cid])
        
    def _changePlot(self, xpix, ypix):
        data = self.data[:, ypix, xpix, self.cid]
        power = self.power[:, ypix, xpix, self.cid]
        power /= power.max()
        self.timeseries.set_ydata(data)
        self.axTS.set_ylim(data.min(), data.max())
        self.powerSpectrum.set_ydata(self.power[:, ypix, xpix, self.cid])
        self.point.set_offsets([self.x, self.y])
        
    def _changeRaster(self):
        self.imRaster.set_data(self.data[self.tpix, :, :, self.cid])
        
    def _transposedPosition(self, x, y, t):
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
        wave = Wavelet(self.data[:, ypix, xpix, self.cid],
                       self.dt, **self.kwargs)
        wpower = wave.power/wave.power.max()
        self.contour = self.axWavelet.contourf(self.timei, wave.period/60,
                                               wpower, len(self.levels),
                                               colors=['w'])
        self.contourIm = self.axWavelet.contourf(self.contour,
                                                 levels=self.levels
                                                 )
        self.axWavelet.fill_between(self.timei, wave.coi/60,
                                    wave.period.max()/60, color='grey',
                                    alpha=0.4, hatch='x')
        self.axWavelet.set_title('Wavelet Power Spectrum')
        self.axWavelet.set_xlabel('Time (sec)')
        self.axWavelet.set_ylabel('Period (minute)')
        self.axWavelet.set_xlim(self.timei[0], self.timei[-1])
        self.axWavelet.set_yscale('log', basey=2)
        self.axWavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axWavelet.ticklabel_format(axis='y',style='plain')
        self.vlineWavelet = self.axWavelet.axvline(self.t,
                                                   ls='dashed',
                                                   color='k')
        self.axWavelet.set_ylim(16, 0.5)
        
def _isoRefTime(refTime):
    year = refTime[:4]
    month = refTime[4:6]
    day = refTime[6:8]
    hour = refTime[9:11]
    minute = refTime[11:13]
    sec = refTime[13:15]
    isot = '%s-%s-%sT%s:%s:%s'%(year, month, day, hour, minute, sec)
    return Time(isot)