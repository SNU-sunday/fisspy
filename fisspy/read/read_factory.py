"""
"""

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
from fisspy.read.readbase import _getRaster, _getHeader, _readFrame

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ["rawData","FISS"]

class rawData:
    """
    rawData class. Used to read a raw data of the FISS.
    
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
        return _getRaster(self.data, self.wave, wv, self.wvDelt, hw=hw)
    
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
    FISS class. Used to read a FISS data file (proc or comp).
    
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
        
        self.header = _getHeader(file)
        self.pfile = self.header.pop('pfile',False)
        self.data = _readFrame(file, self.pfile, ncoeff=ncoeff)
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
        return _getRaster(self.data, self.wave, wv, self.wvDelt, hw=hw)

    
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
                    wvinput= True, shift2velocity= False):
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
        
class FD:
    """
    """
    def __init__(self, file):
       if file.find('FD') != -1:
           self.ftype = 'FD'