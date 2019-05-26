from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fisspy.read.readbase import _getRaster

__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"


class sigleBand:
    """
    """
    
    def __init__(self, fiss, x=None, y=None, wv=None, **kwargs):
        
        self.xpix = round((x-fiss.xDelt/2)/fiss.xDelt)
        self.x = self.xpix*fiss.xDelt+fiss.xDelt/2
        self.ypix = round((y-fiss.yDelt/2)/fiss.yDelt)
        self.y = self.ypix*fiss.yDelt+fiss.yDelt/2
        self.wvpix = round((wv-fiss.waveCenter)/fiss.wvDelt)
        self.wv = self.wvpix*fiss.wvDelt + fiss.waveCenter
        self.waveCenter = fiss.waveCenter
        self.xDelt = fiss.xDelt
        self.yDelt = fiss.yDelt
        self.wvDelt = fiss.wvDelt
        self.wave = fiss.wave
        self.data = fiss.data
        self.ftype = fiss.ftype
        self.nx = fiss.nx
        self.ny = fiss.ny
        self.nwv = fiss.nwv
        fiss.x = self.x
        fiss.xpix = self.xpix
        fiss.y = self.y
        fiss.ypix = self.ypix
        fiss.wv = self.wv
        fiss.wvpix = self.wvpix
        
        #figure setting
        figsize = kwargs.pop('figsize', [10, 6])
        self.cmap = kwargs.pop('cmap', self.cmap)
        self.imInterp = kwargs.pop('interpolation', 'bilinear')
        self.fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3)
        self.axRaster = self.fig.add_subplot(gs[:, 0])
        self.axSpectro = self.fig.add_subplot(gs[0, 1:])
        self.axProfile = self.fig.add_subplot(gs[1, 1:])
        self.axRaster.set_xlabel('X (arcsec)')
        self.axRaster.set_ylabel('Y (arcsec)')
        self.axSpectro.set_xlabel(r'Wavelength ($\AA$)')
        self.axSpectro.set_ylabel('Y (arcsec)')
        self.axProfile.set_xlabel(r'Wavelength ($\AA$)')
        self.axProfile.set_ylabel('Intensity (Count)')
        self.axRaster.set_title(fiss.date)
        self.axSpectro.set_title("X = %.2f'', Y = %.2f''"%(self.x, self.y))
        self.axRaster.set_xlim(fiss.extentRaster[0], fiss.extentRaster[1])
        self.axRaster.set_ylim(fiss.extentRaster[2], fiss.extentRaster[3])
        self.axSpectro.set_xlim(fiss.extentSpectro[0], fiss.extentSpectro[1])
        self.axSpectro.set_ylim(fiss.extentSpectro[2], fiss.extentSpectro[3])
        self.axProfile.set_title(fiss.band)
        self.axProfile.set_xlim(fiss.wave.min(), fiss.wave.max())
        self.axProfile.set_ylim(self.data.min()-100, self.data.max()+100)
        self.axProfile.tick_params(direction='in')
        
        # Draw
        self.imRaster = self.axRaster.imshow(self.data[:,:,self.wvpix].T,
                                             fiss.cmap,
                                             origin='lower',
                                             extent=fiss.extentRaster,
                                             interpolation=fiss.imInterp,
                                             **kwargs)
        self.imSpectro = self.axSpectro.imshow(self.data[self.xpix],
                                               fiss.cmap,
                                               origin='lower',
                                               extent=fiss.extentSpectro,
                                               **kwargs)
        self.plotProfile = self.axProfile.plot(self.wave,
                                               self.data[self.x, self.y],
                                               color='k')[0]
        self.vlineRaster = self.axRaster.vlines(self.x,
                                                fiss.extentRaster[2],
                                                fiss.extentRaster[3],
                                                linestyles='dashed')
        self.pointRaster = self.axRaster.scatter(self.x, self.y, 50,
                                                 marker='x',
                                                 color='k')
        self.vlineSpectro = self.axSpectro.vlines(self.wv,
                                                  fiss.extentSpectro[2],
                                                  fiss.extentSpectro[3],
                                                  linestyles='dashed')
        self.hlineSpectro = self.axSpectro.hlines(self.y,
                                                  fiss.extentSpectro[0],
                                                  fiss.extentSpectro[1],
                                                  linestyles='dashed')
        self.hlineProfile = self.axProfile.hlines(self.wv,
                                                  fiss.extentSpectro[2],
                                                  fiss.extentSpectro[3],
                                                  linestyles='dashed',
                                                  colors='b')
        self.fig.tight_layout()
        self.fig.canvas.draw('key_press_event', self._on_key)
        
    def _on_key(self, event):
        
        # Interactive Raster
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
            if self.wvpix < self.nwv-1:
                self.wvpix += 1
            else:
                self.wvpix = 0
        elif event.key == 'left':
            if self.wvpix > 0:
                self.wvpix -= 1
            else:
                self.wvpix = self.nwv-1
                
        self.x = self.xpix*self.xDelt+self.xDelt/2
        self.y = self.ypix*self.yDelt+self.yDelt/2
        self.wv = self.wvpix*self.wvDelt+self.waveCenter
        
        if event.key == 'ctrl+'

#def singleBand(fiss, x=None, y=None, wv=None, **kwargs):
#    """
#    Draw the interactive image for single band FISS data.
#    """
#    figsize = kwargs.pop('figsize', [10, 8])
#    fiss.cmap = kwargs.pop('cmap', fiss.cmap)
#    fiss.imInterp = kwargs.pop('interpolation', 'bilinear')
#    fiss.singlefig = plt.figure(figsize=figsize)
#    if not x:
#        x = fiss.nx//2*fiss.scale
#    if not y:
#        y = fiss.ny//2*fiss.scale
#    if not wv:
#        wv = fiss.nwv//2
#    fiss.x = round(x/fiss.scale)  # position of x in pixel value
#    fiss.y = round(y/fiss.scale)  # position of y in pixel value
#    fiss.wv = round(wv/fiss.wvscale)  # wavelength in pixel value
#    gs = gridspec.GridSpec(2, 3)
#    fiss.axRaster = fiss.fig.add_subplot(gs[:, 0])
#    fiss.axSpectrogram = fiss.fig.add_subplot(gs[0, 1:])
#    fiss.axProfile = fiss.fig.add_subplot(gs[1, 1:])
#    if fiss.ftype == 'proc' or fiss.ftype == 'comp':
#        fiss.imRaster = fiss.axRaster.imshow(fiss.data[:,:,fiss.wv], 
#                                             fiss.cmap,
#                                             origin='lower',
#                                             extent=fiss.extentRaster,
#                                             interpolation = fiss.imInterp,
#                                             **kwargs)
#        fiss.imSpectro = fiss.axSpectrogram.imshow(fiss.data[:,fiss.x],
#                                                   fiss.cmap,
#                                                   origin='lower',
#                                                   extent=fiss.extentSpectro,
#                                                   interpolation = fiss.imInterp,
#                                                   **kwargs)
#        fiss.plotProfile = fiss.axProfile.plot(fiss.wave,
#                                               fiss.data[fiss.y, fiss.x],
#                                               color='k')[0]
#
#    elif fiss.ftype == 'raw':
#        fiss.imRaster = fiss.axRaster.imshow(fiss.data[:,:,fiss.wv].T, 
#                                             fiss.cmap,
#                                             origin='lower',
#                                             extent=fiss.extentRaster,
#                                             interpolation = fiss.imInterp,
#                                             **kwargs)
#        fiss.imSpectro = fiss.axSpectrogram.imshow(fiss.data[fiss.x],
#                                                   fiss.cmap,
#                                                   origin='lower',
#                                                   extent=fiss.extentSpectro,
#                                                   **kwargs)
#        fiss.plotProfile = fiss.axProfile.plot(fiss.wave,
#                                               fiss.data[fiss.x, fiss.y],
#                                               color='k')[0]
#        
#    fiss.axRaster.set_xlabel('X (arcsec)')
#    fiss.axRaster.set_ylabel('Y (arcsec)')
#    fiss.axSpectro.set_ylabel('Y (arcsec)')
#    fiss.axSpectro.set_xlabel(r'Wavelength ($\AA$)')
#    fiss.axProfile.set_xlabel(r'Wavelength ($\AA$)')
#    fiss.axProfile.set_ylabel('Intensitiy (Count)')
#    fiss.axRaster.set_title(fiss.date)
#    fiss.axProfile.set_title("X=%.2f'', Y=%.2f''"%(x, y))
#    fiss.rasterVline = fiss.axRaster.vlines()
    
    
    