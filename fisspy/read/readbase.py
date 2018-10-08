"""
"""

from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve as conv
from os.path import dirname, basename, join
from fisspy import cm
import matplotlib.pyplot as plt
from astropy.constants import c
from fisspy.analysis.doppler import lambdameter

__author__= "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"

class FISS(object):
    """
    FISS(file)
    
    FISS class. Used to read a FISS data file.
    
    Examples
    --------
    >>> from fisspy.read import FISS
    >>> import fisspy.data.sample
    >>> fiss = FISS(fisspy.data.sample.FISS_IMAGE)
    """
    
    def __init__(self, file, noiseSuppresion= False,
                 simpleWvCalib= True, absScale= True, **kwargs):
        if file.find('Flat') != -1:
            self.ftype = 'Raw Flat'
        elif file.find('FLAT') != -1:
            self.ftype = 'Master Flat'
        elif file.find('SLIT') != -1:
            self.ftype = 'Slit Pattern'
        elif file.find('Cal') != -1:
            self.ftype = 'Calibration File'
        elif file.find('BiasDark') != -1:
            self.ftype = 'BiasDark'
        elif file.find('1.fts') != -1:
            self.ftype = 'Processed Data'
        elif file.find('c.fts') != -1:
            self.ftype = 'Compressed Data'
        elif file.find('p.fts') != -1:
            self.ftype = 'PCA Components'
        elif file.find('mask') != -1:
            self.ftype = 'FISS Data Mask'
        elif file.find('t.fts') != -1:
            self.ftype = 'FISS Data Time'
        elif file.find('FD') != -1 and (file.find('A.fts') != -1 or
                       file.find('B.fts') !=- 1):
            self.ftype = 'FISS Data'
        elif (file.find('FD') == -1 and file.find('FLAT') == -1 and
              file.find('SLIT') == -1 and (file.find('A.fts') != -1 or
                                            file.find('B.fts') != -1)):
            self.ftype = 'Raw Data'
            
        self.filename = file
        self.dirname = dirname(file)
        self.basename = basename(file)
        self.header = self._getHeader()
        self.pfile = self.header.pop('pfile',False)
        self.ndim = self.header['naxis']
        
        if self.ftype == 'Processed Data' or self.ftype == 'Compressed Data':
            self.ny = self.header['naxis2']
            self.nx = self.header['naxis3']
            self.nwv = self.header['naxis1']
            self.date = self.header['date']
            self.band = self.header['wavelen'][:4]
            self.frame = self._readFrame()
            self.refProfile = self.frame.mean((0,1))
            self.wv = self._waveCalibration(simpleWvCalib= simpleWvCalib,
                                            absScale= absScale, **kwargs)
            self.wvRef = self.centralWavelength = self.header['crval1']
            
            self.noiseSuppression = False
            
            if noiseSuppresion:
                self._noiseSuppresion()
            
            if self.band == '6562':
                self.camera = 'A'
                self.set = '1'
                self.cm = cm.ha
            elif self.band == '8542':
                self.camera = 'B'
                self.set = '1'
                self.cm = cm.ca
            elif self.band == '5889':
                self.camera = 'A'
                self.set = '2'
                self.cm = cm.na
            elif self.band == '5434':
                self.camera = 'B'
                self.set = '2'
                self.cm = cm.fe
        elif self.ftype == 'FISS Data':
            self.cam = file[-5]
            self.band = str(self.header['wvrest0'])[:4]
            self.time = fits.getdata(join(self.dirname,
                                          file.replace(self.cam, 't')))
            self.refTime = self.header['reftime']
            self.frame = fits.getdata(file)
            if self.band == '6562':
                self.camera = 'A'
                self.set = '1'
                self.cm = [cm.ha, cm.ha, plt.cm.jet, cm.ha, plt.cm.jet]
            elif self.band == '8542':
                self.camera = 'B'
                self.set = '1'
                self.cm = [cm.ca, cm.ca, plt.cm.jet, cm.ca, plt.cm.jet]
            elif self.band == '5889':
                self.camera = 'A'
                self.set = '2'
                self.cm = [cm.na, cm.na, plt.cm.jet, cm.na, plt.cm.jet]
            elif self.band == '5434':
                self.camera = 'B'
                self.set = '2'
                self.cm = [cm.fe, cm.fe, plt.cm.jet, cm.fe, plt.cm.jet]
        else:
            self.cm = plt.cm.gray
            self.frame = fits.getdata(file)
        
        
    def _getHeader(self):
        """
        _getHeader(self)
        
        Get the FISS fts file header.
        
        Returns
        -------
        header : astropy.io.fits.Header
            The fts file header.
        
        Notes
        -----
            This function automatically check the existance of the pca file by
            reading the fts header.
        
        Example
        -------
        """
        header0 = fits.getheader(self.filename)
    
        pfile = header0.pop('pfile',False)
        if not pfile:
            return header0
        else:
            header = fits.Header()
            header['pfile']=pfile
            for i in header0['comment']:
                sori = i.split('=')
                if len(sori) == 1:
                    skv = sori[0].split(None,1)
                    if len(skv) == 1:
                        pass
                    else:
                        header[skv[0]] = skv[1]
                else:
                    key = sori[0]
                    svc = sori[1].split('/')
                    try:
                        item = float(svc[0])
                    except:
                        item = svc[0].split("'")
                        if len(item) != 1:
                            item = item[1].split(None,0)[0]
                        else:
                            item = item[0].split(None,0)[0]
                    try:
                        if item-int(svc[0]) == 0:
                            item = int(item)
                    except:
                        pass
                    if len(svc) == 1:
                        header[key] = item
                    else:
                        header[key] = (item,svc[1])
                        
        header['simple'] = True
        alignl=header0.pop('alignl',-1)
        
        if alignl == 0:
            keys=['reflect','reffr','reffi','cdelt2','cdelt3','crota2',
                  'crpix3','shift3','crpix2','shift2','margin2','margin3']
            header['alignl'] = (alignl,'Alignment level')
            for i in keys:
                header[i] = (header0[i],header0.comments[i])
            header['history'] = str(header0['history'])
        if alignl == 1:
            keys=['reflect','reffr','reffi','cdelt2','cdelt3','crota1',
                  'crota2','crpix3','crval3','shift3','crpix2','crval2',
                  'shift2','margin2','margin3']
            header['alignl'] = (alignl,'Alignment level')
            for i in keys:
                header[i] = (header0[i],header0.comments[i])
            header['history'] = str(header0['history'])
            
        return header
        
    def _readFrame(self):
        """
        _readFrame(self):
            
        Read the FISS fts file.
        
        Example
        -------
        """
        
        if not self.pfile:
            spec = fits.getdata(self.filename)
        else:
            spec = self._readPCA()
            
        spec = spec.transpose((1,0,2)).astype(float)
        return spec
            
    def _readPCA(self, ncoeff=False):
        """
        """
        
        pdata = fits.getdata(join(self.dirname, self.pfile))
        data = fits.getdata(self.filename)
        ncoeff1 = data.shape[2] - 1
        if not ncoeff:
            ncoeff = ncoeff1
        elif ncoeff > ncoeff1:
            ncoeff = ncoeff1
            
        spec = np.dot(data[:,:,0:ncoeff], pdata[0:ncoeff,:])
        spec *= 10.**data[:,:,ncoeff][:,:,None]
        return spec
    
    def _waveCalibration(self, simpleWvCalib= True, absScale= True,
                         **kwargs):
        """
        """
        method = kwargs.pop('method', True)
        if simpleWvCalib:
            if absScale:
                return (np.arange(self.nwv) -
                        self.header['crpix1']) * self.header['cdelt1'] + self.header['crval1']
            else:
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
    
    def _noiseSuppresion(self, **kwargs):
        """
        """
        windowLength = kwargs.pop('window_length', 7)
        polyOrder = kwargs.pop('polyorder', 2)
        deriv = kwargs.pop('deriv', 0)
        delta = kwargs.pop('delta', 1.0)
        mode = kwargs.pop('mode', 'interp')
        cval = kwargs.pop('cval', 0.0)
        
        self.frame = savgol_filter(self.frame, windowLength, polyOrder,
                                   deriv= deriv, delta= delta, cval= cval,
                                   mode= mode)
        self.noiseSuppression = True
    
    def getRaster(self, wv, hw= 0.05, noReturn=True):
        """
        getRaster(wv, hw)
        
        Make raster images for given wavelengths with in width 2*hw
        
        Parameters
        ----------
        wv   : float or 1d ndarray
            Referenced wavelengths.
        hw   : float
            A half-width of wavelength integration in unit of Angstrom.
            Default is 0.05
            
        Example
        -------
        
        """
        wv = np.array(wv).flatten()
        dldw = self.header['cdelt1']
        
        if hw < abs(dldw)/2.:
            hw = abs(dldw)/2.
            
        s = np.abs(self.wv - wv[:, None]) <= hw
        self.raster = np.array([self.frame[:,:,i].sum(2)/i.sum() for i in s])
        self.rasterWavelength = wv
        if not noReturn:
            return self.raster
        
    def showRaster(self, **kwargs):
        """
        """
        nRaster = self.raster.shape[0]
        figsize = kwargs.get('figsize', [2+nRaster*2.5, 6])
        fig, ax = plt.subplots(1, nRaster, figsize= figsize, **kwargs)
        if nRaster == 1:
            ax = [ax]
        for n, raster in enumerate(self.raster):
            ax[n].imshow(raster, self.cm, origin='lower', **kwargs)
            ax[n].set_title(r'%.2f $\AA$'%self.rasterWavelength[n])
            ax[n].set_xlabel('X [pixel]')
            ax[n].set_ylabel('Y [pixel]')
            
        fig.tight_layout(w_pad=0)          
        fig.subplots_adjust(top=0.9)
        fig.suptitle(r'GST/FISS %s $\AA$ Band %s'%(self.band, self.date),
                     weight='bold')
        fig.show()
        
    def plotRefProfile(self, **kwargs):
        """
        """
        figsize = kwargs.get('figsize', [8,6])
        color = kwargs.get('color', 'k')
        samefig = kwargs.pop('samefig', False)
        if not samefig:
            plt.figure(figsize= figsize)
            plt.title('GST/FISS %s Band %s'%(self.band, self.date))
            plt.xlabel(r'Wavelength [$\AA$]')
            plt.ylabel(r'Intensity [Count]')
            plt.minorticks_on()
            plt.tick_params(which='major', direction='in', width= 1.5, size=5)
            plt.tick_params(which='minor', direction='in', size=3)
            plt.xlim(self.wv.min(), self.wv.max())
        plt.plot(self.wv, self.refProfile, color=color)
        plt.tight_layout()
        plt.show()
        
        
    def plotProfile(self, x, y, **kwargs):
        """
        """
        figsize = kwargs.get('figsize', [8,6])
        color = kwargs.get('color', 'k')
        samefig = kwargs.pop('samefig', False)
        if not samefig:
            plt.figure(figsize= figsize)
            plt.title('GST/FISS %s Band %s'%(self.band, self.date))
            plt.xlabel(r'Wavelength [$\AA$]')
            plt.ylabel(r'Intensity [Count]')
            plt.minorticks_on()
            plt.tick_params(which='major', direction='in', width= 1.5, size=5)
            plt.tick_params(which='minor', direction='in', size=3)
            plt.xlim(self.wv.min(), self.wv.max())
            
        plt.plot(self.wv, self.frame[y,x], color = color)
        plt.tight_layout()
        plt.show()
        
    def showSpectrograph(self, x, **kwargs):
        """
        """
        figsize = kwargs.get('figsize', [8,6])
        plt.figure(figsize= figsize)
        plt.title('GST/FISS %s Band %s'%(self.band, self.date))
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel(r'Slit [pix]')
        plt.tick_params(which='major',width= 1.5, size= 5)
        plt.tight_layout()
        plt.show()
        
    
    def lambdaMeter(self, hw= 0.03, sp= 5e3, wvRange= False,
                    wvinput= True, shift2velocity= False):
        """
        """
        lineShift, intensity = lambdameter(self.wv, self.frame,
                                           ref_spectrum= self.refProfile,
                                           wvRange= wvRange, hw= hw,
                                           wvinput= wvinput)
        
        if shift2velocity:
            LOSvelocity = lineShift * c.to('km/s').value/self.centralWavelength
            return LOSvelocity, intensity
        else:
            return lineShift, intensity
        
    def showRawData(self, frameNumber= False, axis= 0, **kwargs):
        """
        """
        figsize = kwargs.get('figsize', [8,6])
        plt.figure(figsize= figsize)
        if self.ndim == 2:
            plt.imshow(self.frame, self.cm, origin='lower')
        elif self.ndim ==3:
            if axis == 0:
                plt.imshow(self.frame[frameNumber],
                           self.cm, origin='lower')
            elif axis == 1:
                plt.imshow(self.frame[:,frameNumber],
                           self.cm, origin='lower')
            elif axis == 2:
                plt.imshow(self.frame[:,:,frameNumber],
                           self.cm, origin='lower')
        plt.title(self.ftype)
        plt.xlabel('X [pix]')
        plt.ylabel('Y [pix]')
        plt.tight_layout()
        plt.show()
        
    def plotTimeseries(self, position, wavelegthFrame, **kwargs):
        """
        """
        figsize = kwargs.get('figsize', [8,6])
        plt.figure(figsize= figsize)
        plt.plot(self.time, self.frame[:,position[1],
                                       position[0],wavelegthFrame])
        plt.title('X = %i, Y = %i / Wavelegth = %s'%(position[0],
                                                     position[1],
                                                     self.header['ID%s'%wavelegthFrame]))
        plt.xlabel('Time [min]')
        plt.ylabel(r'Velocity [km s$^{-1}$]')
        plt.minorticks_on()
        plt.tick_params(which='major', direction='in', width= 1.5, size=5)
        plt.tick_params(which='minor', direction='in', size=3)
        plt.xlim(self.time[0], self.time[-1])
        plt.tight_layout()
        plt.show()
        
    def showFD(self, Timeframe, **kwargs):
        """
        """
        
        figsize = kwargs.get('figsize', [self.frame.shape[2]/100*5+2,
                                         self.frame.shape[1]/100+2])
        fig, ax = plt.subplots(1, 5, figsize= figsize)
        im = [None]*5
        for i in range(5):
            im[i] = ax[i].imshow(self.frame[Timeframe,:,:,i], self.cm[i],
                                  origin='lower')
            ax[i].set_title(self.header['ID%i'%i])
            ax[i].set_xlabel('X [pix]')
            ax[i].set_ylabel('Y [pix]')
        
        im[2].set_clim(-4,4)
        im[4].set_clim(-1.5,1.5)
        fig.tight_layout(w_pad=0)
        fig.show()