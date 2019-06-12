from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fisspy.read.readbase import getRaster as _getRaster
from fisspy.image.base import alignoffset, shift3d

__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"


class singleBand:
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
    
    def __init__(self, fiss, x=None, y=None, wv=None, scale='minMax', 
                 sigFactor=3, helpBox=True, **kwargs):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        
        if not x:
            x = fiss.nx//2*fiss.xDelt
        if not y:
            y = fiss.ny//2*fiss.yDelt
        if not wv:
            wv = fiss.centralWavelength
        plt.rcParams['keymap.back'].remove('left')
        plt.rcParams['keymap.forward'].remove('right')
        self.scale = scale
        self.sigFactor = sigFactor
        self.hw = kwargs.pop('hw', 0.05)
        self.xpix = round((x-fiss.xDelt/2)/fiss.xDelt)
        self.x = self.xpix*fiss.xDelt+fiss.xDelt/2
        self.ypix = round((y-fiss.yDelt/2)/fiss.yDelt)
        self.y = self.ypix*fiss.yDelt+fiss.yDelt/2
        self.wv = wv
        self.xpix0 = self.xpix
        self.ypix0 = self.ypix
        self.x0 = self.x
        self.y0 = self.y
        self.wv0 = self.wv
        self.xpixH = self.xpix
        self.ypixH = self.ypix
        self.xH = self.x
        self.yH = self.y
        self.wvH = self.wv
        self.centralWavelength = fiss.centralWavelength
        self.xDelt = fiss.xDelt
        self.yDelt = fiss.yDelt
        self.wvDelt = fiss.wvDelt
        self.wave = fiss.wave
        self.data = fiss.data
        self.ftype = fiss.ftype
        self.nx = fiss.nx
        self.ny = fiss.ny
        self.nwv = fiss.nwv
        self.band = fiss.band
        self.cam = fiss.cam
        fiss.x = self.x
        fiss.xpix = self.xpix
        fiss.y = self.y
        fiss.ypix = self.ypix
        fiss.wv = self.wv

            
        #Keyboard helpBox
        if helpBox:
            helpFig = plt.figure('Keyboard Help Box', figsize=[3.5,3])
            ax = helpFig.add_subplot(111)
            ax.set_position([0,0,1,1])
            ax.set_axis_off()
            ax.text(0.05,0.9,'ctrl+h: Reset to original setting')
            ax.text(0.05,0.8,'ctrl+right: Move to right')
            ax.text(0.05,0.7,'ctrl+left: Move to left')
            ax.text(0.05,0.6,'ctrl+up: Move to up')
            ax.text(0.05,0.5,'ctrl+down: Move to down')
            ax.text(0.05,0.4,'right: Increase the wavelength')
            ax.text(0.05,0.3,'left: Decrease the wavelength')
            ax.text(0.05,0.2,'spacebar: Change to current mouse point')
            
            
        #figure setting
        figsize = kwargs.pop('figsize', [10, 6])
        self.cmap = kwargs.pop('cmap', fiss.cmap)
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.set_window_title(self.band)
        self.imInterp = kwargs.get('interpolation', fiss.imInterp)
        gs = gridspec.GridSpec(2, 3)
        self.axRaster = self.fig.add_subplot(gs[:, 0])
        self.axSpectro = self.fig.add_subplot(gs[0, 1:])
        self.axProfile = self.fig.add_subplot(gs[1, 1:])
        fiss.axRaster = self.axRaster
        fiss.axSpectro = self.axSpectro
        fiss.axProfile = self.axProfile
        self.axRaster.set_xlabel('X (arcsec)')
        self.axRaster.set_ylabel('Y (arcsec)')
        self.axSpectro.set_xlabel(r'Wavelength ($\AA$)')
        self.axSpectro.set_ylabel('Y (arcsec)')
        self.axProfile.set_xlabel(r'Wavelength ($\AA$)')
        self.axProfile.set_ylabel('Intensity (Count)')
        self.axRaster.set_title(fiss.date)
        self.axSpectro.set_title(r"X = %.2f'', Y = %.2f'' (X$_{pix}$ = %i, Y$_{pix}$ = %i)"%(self.x, self.y, self.xpix, self.ypix))
        self.axRaster.set_xlim(fiss.extentRaster[0], fiss.extentRaster[1])
        self.axRaster.set_ylim(fiss.extentRaster[2], fiss.extentRaster[3])
        self.axSpectro.set_xlim(fiss.extentSpectro[0], fiss.extentSpectro[1])
        self.axSpectro.set_ylim(fiss.extentSpectro[2], fiss.extentSpectro[3])
        self.axProfile.set_title(r'%s Band (wv = %.2f $\AA$)'%(fiss.band, self.wv))
        self.axProfile.set_xlim(fiss.wave.min(), fiss.wave.max())
        self.axProfile.set_ylim(self.data[self.ypix, self.xpix].min()-100,
                                self.data[self.ypix, self.xpix].max()+100)
        self.axProfile.minorticks_on()
        self.axProfile.tick_params(which='both', direction='in')
        
        
        # Draw
        raster = _getRaster(self.data, self.wave, self.wv, self.wvDelt,
                            hw=self.hw)
            
        if self.cam == 'A':
            spectro = self.data[:, self.xpix]
        elif self.cam == 'B':
            spectro = self.data[:, self.xpix,::-1]
        if self.scale == 'log':
            raster = np.log10(raster)
            spectro = np.log10(spectro)
        self.imRaster = self.axRaster.imshow(raster,
                                             fiss.cmap,
                                             origin='lower',
                                             extent=fiss.extentRaster,
                                             **kwargs)
        self.imSpectro = self.axSpectro.imshow(spectro,
                                               fiss.cmap,
                                               origin='lower',
                                               extent=fiss.extentSpectro,
                                               **kwargs)
        self.plotProfile = self.axProfile.plot(self.wave,
                                               self.data[self.ypix, self.xpix],
                                               color='k')[0]
        
        if self.scale == 'std':
            self.imRaster.set_clim(np.median(raster)-raster.std()*self.sigFactor,
                                   np.median(raster)+raster.std()*self.sigFactor)
            self.imSpectro.set_clim(np.median(spectro)-spectro.std()*self.sigFactor,
                                    np.median(spectro)+spectro.std()*self.sigFactor)
        else:
            self.imRaster.set_clim(raster.min(), raster.max())
            self.imSpectro.set_clim(spectro.min(), spectro.max())
        
        # Reference
        self.vlineRaster = self.axRaster.axvline(self.x,
                                                 linestyle='dashed',
                                                 color='lime')
        self.vlineProfile = self.axProfile.axvline(self.wv,
                                                   ls='dashed',
                                                   c='b')
        self.vlineSpectro = self.axSpectro.axvline(self.wv,
                                                   ls='dashed',
                                                   c='lime')
        self.hlineSpectro = self.axSpectro.axhline(self.y,
                                                   ls='dashed',
                                                   c='lime')
        self.pointRaster = self.axRaster.scatter(self.x, self.y, 50,
                                                 marker='x',
                                                 color='r')
        self.axSpectro.set_aspect(adjustable='box', aspect='auto')
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.show()
        
    def _on_key(self, event):
        
        ### Interactive keyboard input
        # Position
        if event.key == 'ctrl+right':
            if self.xpix < self.nx-1:
                self.xpix += 1
            else:
                self.xpix = 0
        elif event.key == 'ctrl+left':
            if self.xpix > 0:
                self.xpix -= 1
            else:
                self.xpix = self.nx-1
        elif event.key == 'ctrl+up':
            if self.ypix < self.ny-1:
                self.ypix += 1
            else:
                self.ypix = 0
        elif event.key == 'ctrl+down':
            if self.ypix > 0:
                self.ypix -= 1
            else:
                self.ypix = self.ny-1
        elif event.key == 'right':
            if self.wv < self.wave.max():
                self.wv += abs(self.wvDelt)
            else:
                self.wv = self.wave.min()
        elif event.key == 'left':
            if self.wv > self.wave.min():
                self.wv -= abs(self.wvDelt)
            else:
                self.wv = self.wave.max()
        elif event.key == ' ' and event.inaxes == self.axRaster:
            self.x = event.xdata
            self.y = event.ydata
        elif event.key == ' ' and event.inaxes == self.axProfile:
            self.wv = event.xdata
        elif event.key == ' ' and event.inaxes == self.axSpectro:
            self.wv = event.xdata
            self.y = event.ydata
        elif event.key == 'ctrl+h':
            self.wv = self.wvH
            self.x = self.xH
            self.xpix = self.xpixH
            self.y = self.yH
            self.ypix = self.ypixH
        
        if self.x != self.x0 or self.y != self.y0:
            self.xpix = int(round((self.x-self.xDelt/2)/self.xDelt))
            self.ypix = int(round((self.y-self.yDelt/2)/self.yDelt))
        if self.xpix != self.xpix0 or self.ypix != self.ypix0:
            self.x = self.xpix*self.xDelt+self.xDelt/2
            self.y = self.ypix*self.yDelt+self.yDelt/2
            self._chSpect()
        if self.wv != self.wv0:
            self._chRaster()
        self.fig.canvas.draw_idle()
        
        
    def _chRaster(self):
        self.wv0 = self.wv
        raster = _getRaster(self.data, self.wave, self.wv, self.wvDelt,
                             hw=self.hw)
        if self.scale == 'log':
            raster = np.log10(raster)
        self.imRaster.set_data(raster)
        self.vlineProfile.set_xdata(self.wv)
        self.vlineSpectro.set_xdata(self.wv)
        self.axProfile.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.band, self.wv))
        if self.scale == 'std':
            self.imRaster.set_clim(np.median(raster)-raster.std()*self.sigFactor,
                                   np.median(raster)+raster.std()*self.sigFactor)
        else:
            self.imRaster.set_clim(raster.min(), raster.max())
            
    def _chSpect(self):
        self.x0 = self.x
        self.xpix0 = self.xpix
        self.y0 = self.y
        self.ypix0 = self.ypix
        
        if self.cam == 'A':
            spectro = self.data[:, self.xpix]
        elif self.cam == 'B':
            spectro = self.data[:, self.xpix,::-1]
        if self.scale == 'log':
            spectro = np.log10(spectro)
        self.plotProfile.set_ydata(self.data[self.ypix, self.xpix])
        self.imSpectro.set_data(spectro)
        self.hlineSpectro.set_ydata(self.y)
        self.vlineRaster.set_xdata(self.x)
        self.pointRaster.set_offsets([self.x, self.y])
        
        self.axProfile.set_ylim(self.data[self.ypix, self.xpix].min()-100,
                                self.data[self.ypix, self.xpix].max()+100)
        self.axSpectro.set_title(r"X = %.2f'', Y = %.2f'' (X$_{pix}$ = %i, Y$_{pix}$ = %i)"%(self.x, self.y, self.xpix, self.ypix))
        if self.scale == 'std':
            self.imSpectro.set_clim(np.median(spectro)-spectro.std()*self.sigFactor,
                                    np.median(spectro)+spectro.std()*self.sigFactor)
        else:
            self.imSpectro.set_clim(spectro.min(), spectro.max())

    def chRasterClim(self, cmin, cmax):
        self.imRaster.set_clim(cmin, cmax)
    
    def chSpectroClim(self, cmin, cmax):
        self.imSpectro.set_clim(cmin, cmax)
        
    def chcmap(self, cmap):
        self.imRaster.set_cmap(cmap)
        self.imSpectro.set_cmap(cmap)
        
class dualBand:
    """
    Draw interactive FISS raster, spectrogram and profile for dual band.
    
    Parameters
    ----------
    
    """
    def __init__(self, fissA, fissB, x=None, y=None, wvA=None, wvB=None,
                 scale='minMax', sigFactor=3, helpBox=True, **kwargs):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        
        kwargs['interpolation'] = kwargs.pop('interpolation', 'bilinear')
        plt.rcParams['keymap.back'].remove('left')
        plt.rcParams['keymap.forward'].remove('right')
        self.fissA = fissA
        self.fissB = fissB
        self.nx = self.fissA.nx
        self.xDelt = self.fissA.xDelt
        self.yDelt = self.fissA.yDelt
        if self.fissA.ny > self.fissB.ny:
            self.fissA.data = self.fissA.data[:self.fissB.ny]
            self.ny = self.fissB.ny
            self.extentRaster = self.fissB.extentRaster
        elif fissA.ny < fissB.ny:
            self.fissB.data = self.fissB.data[:self.fissA.ny]
            self.ny = self.fissA.ny
            self.extentRaster = self.fissA.extentRaster
        self._xMin = self.extentRaster[0]
        self._xMax = self.extentRaster[1]
        self._yMin = self.extentRaster[2]
        self._yMax = self.extentRaster[3]
        
        sh = alignoffset(self.fissB.data[:,:,50], self.fissA.data[:,:,-50])
        tmp = shift3d(fissB.data.transpose(2, 0, 1), -sh).transpose(1,2,0)
        self.fissB.data = tmp
        tmp[tmp<10]=1
        del tmp
        
        if not x:
            x = self.nx//2*self.xDelt
        if not y:
            y = self.ny//2*self.yDelt
        if not wvA:
            wvA = self.fissA.centralWavelength
        if not wvB:
            wvB = self.fissB.centralWavelength
        xpix = round((x-self.xDelt/2)/self.xDelt)
        ypix = round((y-self.yDelt/2)/self.yDelt)
        self.x = xpix*self.xDelt+self.xDelt/2
        self.y = ypix*self.yDelt+self.yDelt/2
        self.scale = scale
        self.sigFactor = sigFactor
        self.hw = kwargs.pop('hw', 0.05)
        self.wvA = wvA
        self.wvB = wvB
        self.x0 = self.x
        self.y0 = self.y
        self.wvA0 = self.wvA
        self.wvB0 = self.wvB
        self.xH = self.x
        self.yH = self.y
        self.wvAH = self.wvA
        self.wvBH = self.wvB
        
        #Keyboard helpBox
        if helpBox:
            helpFig = plt.figure('Keyboard Help Box', figsize=[3.5,3])
            ax = helpFig.add_subplot(111)
            ax.set_position([0,0,1,1])
            ax.set_axis_off()
            ax.text(0.05,0.92,'ctrl+h: Reset to original setting')
            ax.text(0.05,0.82,'ctrl+right: Move to right')
            ax.text(0.05,0.72,'ctrl+left: Move to left')
            ax.text(0.05,0.62,'ctrl+up: Move to up')
            ax.text(0.05,0.52,'ctrl+down: Move to down')
            ax.text(0.05,0.42,'right: Increase the wavelength of the fissA')
            ax.text(0.05,0.32,'left: Decrease the wavelength of the fissA')
            ax.text(0.05,0.22,'up: Increase the wavelength of the fissB')
            ax.text(0.05,0.12,'down: Decrease the wavelength of the fissB')
            ax.text(0.05,0.02,'spacebar: Change to current mouse point')
        
        #figure setting
        figsize = kwargs.pop('figsize', [12, 6])
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.set_widnow_title('Dual Band Image')
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        gs = gridspec.GridSpec(2,4)
        self.axRasterA = self.fig.add_subplot(gs[:,0])
        self.axRasterB = self.fig.add_subplot(gs[:,1],
                                              sharex=self.axRasterA,
                                              sharey=self.axRasterA)
        self.axProfileA = self.fig.add_subplot(gs[0,2:])
        self.axProfileB = self.fig.add_subplot(gs[1,2:])
        self.axRasterA.set_xlabel('X (arcsec)')
        self.axRasterA.set_ylabel('Y (arcsec)')
        self.axRasterB.set_xlabel('X (arcsec)')
        self.axRasterB.set_ylabel('Y (arcsec)')
        self.axProfileA.set_xlabel(r'Wavelength ($\AA$)')
        self.axProfileA.set_ylabel('Intensity (Count)')
        self.axProfileB.set_xlabel(r'Wavelength ($\AA$)')
        self.axProfileB.set_ylabel('Intensity (Count)')
        self.axRasterA.set_title(r'%s Band'%self.fissA.band)
        self.axRasterB.set_title(r'%s Band'%self.fissB.band)
        self.axProfileA.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.fissA.band, self.wvA))
        self.axProfileB.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.fissB.band, self.wvB))
        self.axRasterA.set_xlim(self.extentRaster[0], self.extentRaster[1])
        self.axRasterB.set_xlim(self.extentRaster[0], self.extentRaster[1])
        self.axRasterA.set_ylim(self.extentRaster[2], self.extentRaster[3])
        self.axRasterB.set_ylim(self.extentRaster[2], self.extentRaster[3])
        self.axProfileA.set_xlim(self.fissA.wave.min(), self.fissA.wave.max())
        self.axProfileB.set_xlim(self.fissB.wave.min(), self.fissB.wave.max())
        self.axProfileA.set_ylim(self.fissA.data[ypix, xpix].min()-100,
                                 self.fissA.data[ypix, xpix].max()+100)
        self.axProfileB.set_ylim(self.fissB.data[ypix, xpix].min()-100,
                                 self.fissB.data[ypix, xpix].max()+100)
        self.axProfileA.minorticks_on()
        self.axProfileA.tick_params(which='both', direction='in')
        self.axProfileB.minorticks_on()
        self.axProfileB.tick_params(which='both', direction='in')
        
        #Draw
        rasterA = _getRaster(self.fissA.data, self.fissA.wave, self.wvA,
                             self.fissA.wvDelt, hw=self.hw)
        rasterB = _getRaster(self.fissB.data, self.fissB.wave, self.wvB,
                             self.fissB.wvDelt, hw=self.hw)
        wh = rasterB >5
        if self.scale == 'log':
            rasterA = np.log10(rasterA)
            rasterB = np.log10(rasterB)
        cmin = rasterB[wh].min()
        self.imRasterA = self.axRasterA.imshow(rasterA,
                                               self.fissA.cmap,
                                               origin='lower',
                                               extent=self.extentRaster,
                                               **kwargs)
        self.imRasterB = self.axRasterB.imshow(rasterB,
                                               self.fissB.cmap,
                                               origin='lower',
                                               extent=self.extentRaster,
                                               **kwargs)
        self.plotProfileA = self.axProfileA.plot(self.fissA.wave,
                                                 self.fissA.data[ypix, xpix],
                                                 color='k')[0]
        self.plotProfileB = self.axProfileB.plot(self.fissB.wave,
                                                 self.fissB.data[ypix, xpix],
                                                 color='k')[0]
        
        if self.scale == 'std':
            self.imRasterA.set_clim(np.median(rasterA)-rasterA.std()*self.sigFactor,
                                    np.median(rasterA)+rasterA.std()*self.sigFactor)
            self.imRasterB.set_clim(np.median(rasterB)-rasterB.std()*self.sigFactor,
                                    np.median(rasterB)+rasterB.std()*self.sigFactor)
        else:
            self.imRasterA.set_clim(rasterA.min(), rasterA.max())
            self.imRasterB.set_clim(cmin, rasterB.max())
            
        #Reference
        self.vlineRasterA = self.axRasterA.axvline(self.x,
                                                   linestyle='dashed',
                                                   color='lime')
        self.vlineRasterB = self.axRasterB.axvline(self.x,
                                                   linestyle='dashed',
                                                   color='lime')
        self.vlineProfileA = self.axProfileA.axvline(self.wvA,
                                                     ls='dashed',
                                                     c='b')
        self.vlineProfileB = self.axProfileB.axvline(self.wvB,
                                                     ls='dashed',
                                                     c='b')
        self.pointRasterA = self.axRasterA.scatter(self.x, self.y, 50,
                                                   marker='x',
                                                   color='r')
        self.pointRasterB = self.axRasterB.scatter(self.x, self.y, 50,
                                                   marker='x',
                                                   color='r')
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        
        plt.show()
    def _on_key(self, event):
        
        if event.key == 'ctrl+right':
            if self.x < self._xMax:
                self.x += self.xDelt
            else:
                self.x = self._xMin+self.xDelt//2
        elif event.key == 'ctrl+left':
            if self.x > self._xMin:
                self.x -= self.xDelt
            else:
                self.x = self._xMax-self.xDelt//2
        elif event.key == 'ctrl+up':
            if self.y < self._yMax:
                self.y += self.yDelt
            else:
                self.y = self._yMin+self.yDelt//2
        elif event.key == 'ctrl+down':
            if self.y > self._yMin:
                self.y -= self.yDelt
            else:
                self.y = self._yMax-self.yDelt//2
        elif event.key == 'right':
            if self.wvA < self.fissA.wave.max():
                self.wvA += abs(self.fissA.wvDelt)
            else:
                self.wvA = self.fissA.wave.min()
        elif event.key == 'left':
            if self.wvA > self.fissA.wave.min():
                self.wvA -= abs(self.fissA.wvDelt)
            else:
                self.wvA = self.fissA.wave.max()
        elif event.key == 'up':
            if self.wvB < self.fissB.wave.max():
                self.wvB += abs(self.fissB.wvDelt)
            else:
                self.wvB = self.fissB.wave.min()
        elif event.key == 'down':
            if self.wvB > self.fissB.wave.min():
                self.wvB -= abs(self.fissB.wvDelt)
            else:
                self.wvB = self.fissB.wave.max()
        elif event.key == ' ' and (event.inaxes == self.axRasterA or 
                                        event.inaxes == self.axRasterB) :
            self.x = event.xdata
            self.y = event.ydata
        elif event.key == ' ' and event.inaxes == self.axProfileA:
            self.wvA = event.xdata
        elif event.key == ' ' and event.inaxes == self.axProfileB:
            self.wvB = event.xdata
        elif event.key == 'ctrl+h':
            self.wvA = self.wvAH
            self.wvB = self.wvBH
            self.x = self.xH
            self.y = self.yH
            
        if self.x != self.x0 or self.y != self.y0:
            self._chSpect()
        if self.wvA != self.wvA0:
            self._chRasterA()
        if self.wvB != self.wvB0:
            self._chRasterB()
        self.fig.canvas.draw_idle()
        
    def _chSpect(self):
        self.x0 = self.x
        self.y0 = self.y
        xpix = int(round((self.x-self.xDelt/2)/self.xDelt))
        ypix = int(round((self.y-self.yDelt/2)/self.yDelt))
        
        self.plotProfileA.set_ydata(self.fissA.data[ypix, xpix])
        self.plotProfileB.set_ydata(self.fissB.data[ypix, xpix])
        self.pointRasterA.set_offsets([self.x, self.y])
        self.pointRasterB.set_offsets([self.x, self.y])
        self.vlineRasterA.set_xdata(self.x)
        self.vlineRasterB.set_xdata(self.x)
        
        self.axProfileA.set_ylim(self.fissA.data[ypix, xpix].min()-100,
                                 self.fissA.data[ypix, xpix].max()+100)
        self.axProfileB.set_ylim(self.fissB.data[ypix, xpix].min()-100,
                                 self.fissB.data[ypix, xpix].max()+100)
        
    def _chRasterA(self):
        self.wvA0 = self.wvA
        rasterA = _getRaster(self.fissA.data, self.fissA.wave, self.wvA,
                             self.fissA.wvDelt,
                             hw=self.hw)
        if self.scale == 'log':
            rasterA = np.log10(rasterA)
        self.imRasterA.set_data(rasterA)
        self.vlineProfileA.set_xdata(self.wvA)
        self.axProfileA.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.fissA.band,
                                                                self.wvA))
        if self.scale == 'std':
            self.imRasterA.set_clim(np.median(rasterA)-rasterA.std()*self.sigFactor,
                                    np.median(rasterA)+rasterA.std()*self.sigFactor)
        else:
            self.imRasterA.set_clim(rasterA.min(), rasterA.max())
        
    def _chRasterB(self):
        self.wvB0 = self.wvB
        rasterB = _getRaster(self.fissB.data, self.fissB.wave, self.wvB,
                             self.fissB.wvDelt,
                             hw=self.hw)
        wh = rasterB > 5
        if self.scale == 'log':
            rasterB = np.log10(rasterB)
        cmin = rasterB[wh].min()
        self.imRasterB.set_data(rasterB)
        self.vlineProfileB.set_xdata(self.wvB)
        self.axProfileB.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.fissB.band,
                                                                self.wvB))
        if self.scale == 'std':
            self.imRasterB.set_clim(np.median(rasterB)-rasterB.std()*self.sigFactor,
                                    np.median(rasterB)+rasterB.std()*self.sigFactor)
        else:
            self.imRasterB.set_clim(cmin, rasterB.max())    