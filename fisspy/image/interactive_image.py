from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ..read.readbase import getRaster as _getRaster


__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ["singleBand", "dualBand"]

class singleBand:
    """
    Draw interactive FISS raster, spectrogram and profile for single band.

    Parameters
    ----------
    fiss: `fisspy.read.FISS`
        FISS class.
    x: `float`
        X position that you draw a spectral profile.
        Default is image center.
    y: `float`
        Y position that you draw a spectral profile.
        Default is image center.
    wv: `float`
        Wavelength positin that you draw a raster images.
        Default is central wavelength.
    scale: `string`
        Scale method of colarbar limit.
        Default is minMax.
        option: 'minMax', 'std', 'log'
    sigFactor: `float`
        Factor of standard deviation.
        This is worked if scale is set to be 'std'
    helpBox: `bool`
        Show the interacitve key and simple explanation.
        Default is True

    Other Parameters
    ----------------
    **kwargs: `~matplotlib.pyplot` properties
    """

    def __init__(self, fiss, x=None, y=None, wv=None, scale='log', sigFactor=2, helpBox=True, **kwargs):

        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass

        if not x:
            X = fiss.nx//2*fiss.xDelt
        else:
            X = x
        if not y:
            Y = fiss.ny//2*fiss.yDelt
        else:
            Y = y
        if not wv:
            WV = fiss.centralWavelength
        else:
            WV = wv

        self.wave = fiss.wave
        self.data = fiss.data
        self.nx = fiss.nx
        self.ny = fiss.ny
        self.nwv = fiss.nwv
        self.band = fiss.band
        self.cam = fiss.cam
        self.cwv = fiss.centralWavelength
        self.dx = fiss.xDelt
        self.dy = fiss.yDelt
        self.dwv = fiss.wvDelt
        self.extentRaster = fiss.extentRaster
        self.extentSpectro = fiss.extentSpectro
        self.scale = scale
        self.sigFactor = sigFactor
        self.hw = kwargs.pop('hw', 0.05)
        self.xp = int(X/self.dx+0.5)
        self.x = self.xp*self.dx
        self.yp = int(Y/self.dy+0.5)
        self.y = self.yp*self.dy
        self.mwv = self.wave[0]
        self.wv = WV
        self.wvp = int((WV-self.mwv)/self.dwv+0.5)
        self.xp0 = self.xp
        self.yp0 = self.yp
        self.wvp0 = self.wvp
        self.xpH = self.xp
        self.ypH = self.yp
        self.wvpH = self.wvp


        #Keyboard helpBox
        if helpBox:
            helpFig = plt.figure('Keyboard Help Box', figsize=[3.5,3])
            ax = helpFig.add_subplot(111)
            ax.set_position([0,0,1,1])
            ax.set_axis_off()
            ax.text(0.05,0.8,'right: Move to right')
            ax.text(0.05,0.7,'left: Move to left')
            ax.text(0.05,0.6,'up: Move to up')
            ax.text(0.05,0.5,'down: Move to down')
            ax.text(0.05,0.2,'spacebar: Change to current mouse point')
            ax.text(0.05,0.9,'ctrl/cmd+h: Reset to original setting position')
            ax.text(0.05,0.4,'ctrl/cmd+right: Increase the wavelength')
            ax.text(0.05,0.3,'ctrl/cmd+left: Decrease the wavelength')
            ax.text(0.05,0.1,'ctrl/cmd+b: Show previous point')
            helpFig.show()


        #figure setting
        figsize = kwargs.pop('figsize', [15, 9])
        self.cmap = kwargs.pop('cmap', fiss.cmap)
        self.fig = plt.figure(figsize=figsize)
        # self.fig.canvas.set_window_title(self.band)
        kwargs['interpolation'] = kwargs.pop('interpolation', 'bilinear')
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
        self.axSpectro.set_title(r"X = %.2f'', Y = %.2f'' (X$_{pix}$ = %i, Y$_{pix}$ = %i)"%(self.x, self.y, self.xp, self.yp))
        self.axRaster.set_xlim(fiss.extentRaster[0], fiss.extentRaster[1])
        self.axRaster.set_ylim(fiss.extentRaster[2], fiss.extentRaster[3])
        self.axSpectro.set_xlim(fiss.extentSpectro[0], fiss.extentSpectro[1])
        self.axSpectro.set_ylim(fiss.extentSpectro[2], fiss.extentSpectro[3])
        self.axProfile.set_title(r'%s Band (wv = %.2f $\AA$)'%(fiss.band, self.wv))
        self.axProfile.set_xlim(fiss.wave.min(), fiss.wave.max())
        ym = self.data[self.yp, self.xp].min()
        yM = self.data[self.yp, self.xp].max()
        margin = (yM-ym)*0.05
        self.axProfile.set_ylim(ym-margin, yM+margin)
        self.axProfile.minorticks_on()
        self.axProfile.tick_params(which='both', direction='in')


        # Draw
        raster = _getRaster(self.data, self.wave, self.wv, self.dwv, hw=self.hw)
        M = raster.max()
        if M > 1e2:
            m = raster[raster > 1e2].min()
        else:
            m = raster.min()
        if self.cam == 'A':
            spectro = self.data[:, self.xp]
        elif self.cam == 'B':
            spectro = self.data[:, self.xp,::-1]
        if self.scale == 'log':
            raster = np.log10(raster)
            spectro = np.log10(spectro)
            M = np.log10(M)
            m = np.log10(m)
        self.imRaster = self.axRaster.imshow(raster, fiss.cmap, origin='lower', extent=fiss.extentRaster, **kwargs)
        self.imSpectro = self.axSpectro.imshow(spectro, fiss.cmap, origin='lower', extent=fiss.extentSpectro, **kwargs)
        self.plotProfile = self.axProfile.plot(self.wave, self.data[self.yp, self.xp], color='k')[0]

        if self.scale == 'std':
            self.imRaster.set_clim(np.median(raster)-raster.std()*self.sigFactor, np.median(raster)+raster.std()*self.sigFactor)
            self.imSpectro.set_clim(spectro.min(), spectro.max())
        else:
            self.imRaster.set_clim(m, M)
            self.imSpectro.set_clim(spectro.min(), spectro.max())

        # Reference
        self.vlineRaster = self.axRaster.axvline(self.x, linestyle='dashed', color='lime')
        self.vlineProfile = self.axProfile.axvline(self.wv, ls='dashed', c='b')
        self.vlineSpectro = self.axSpectro.axvline(self.wv, ls='dashed', c='lime')
        self.hlineSpectro = self.axSpectro.axhline(self.y, ls='dashed', c='lime')
        self.pointRaster = self.axRaster.scatter(self.x, self.y, 50, marker='x', color='r')
        self.axSpectro.set_aspect(adjustable='box', aspect='auto')
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)

        self.fig.show()

    def _onKey(self, event):
        if event.key == 'right':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            if self.xp < self.nx-1:
                self.xp += 1
            else:
                self.xp = 0
            self.x = self.xp*self.dx
            self._chSpect()
        elif event.key == 'left':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            if self.xp > 0:
                self.xp -= 1
            else:
                self.xp = self.nx-1
            self.x = self.xp*self.dx
            self._chSpect()
        elif event.key == 'up':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            if self.yp < self.ny-1:
                self.yp += 1
            else:
                self.yp = 0
            self.y = self.yp*self.dy
            self._chSpect()
        elif event.key == 'down':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            if self.yp > 0:
                self.yp -= 1
            else:
                self.yp = self.ny-1
            self.y = self.yp*self.dy
            self._chSpect()
        elif event.key == 'ctrl+right' or event.key == 'cmd+right':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            if self.wvp < self.nwv-1:
                self.wvp += 1
            else:
                self.wvp = 0
            self.wv = self.wvp*self.dwv+self.mwv
            self._chRaster()
        elif event.key == 'ctrl+left' or event.key == 'cmd+left':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            if self.wvp > 0:
                self.wvp -= 1
            else:
                self.wvp = self.nwv-1
            self.wv = self.wvp*self.dwv+self.mwv
            self._chRaster()
        elif event.key == ' ' and event.inaxes == self.axRaster:
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            self.x = event.xdata
            self.y = event.ydata
            self.xp = int(self.x/self.dx+0.5)
            self.yp = int(self.y/self.dy+0.5)
            self.x = self.xp*self.dx
            self.y = self.yp*self.dy
            self._chSpect()
        elif event.key == ' ' and event.inaxes == self.axProfile:
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            self.wv = event.xdata
            self.wvp = int((self.wv-self.mwv)/self.dwv+0.5)
            self.wv = self.wvp*self.dwv+self.mwv
            self._chRaster()
        elif event.key == ' ' and event.inaxes == self.axSpectro:
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp0 = self.wvp
            self.wv = event.xdata
            self.wvp = int((self.wv-self.mwv)/self.dwv+0.5)
            self.wv = self.wvp*self.dwv+self.mwv
            self.y = event.ydata
            self.yp = int(self.y/self.dy+0.5)
            self.y = self.yp*self.dy
            self._chRaster()
            self._chSpect()
        elif event.key == 'ctrl+h' or event.key == 'cmd+h':
            self.wvp0 = self.wvp
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvp = self.wvpH
            self.xp = self.xpH
            self.yp = self.ypH
            self.x = self.xp*self.dx
            self.y = self.yp*self.dy
            self.wv = self.wvp*self.dwv+self.mwv
            self._chRaster()
            self._chSpect()
        elif event.key == 'ctrl+b' or event.key == 'cmd+b':
            x = self.xp
            y = self.yp
            wv = self.wvp
            self.xp = self.xp0
            self.yp = self.yp0
            self.wvp = self.wvp0
            self.x = self.xp*self.dx
            self.y = self.yp*self.dy
            self.wv = self.wvp*self.dwv+self.mwv
            self._chRaster()
            self._chSpect()
            self.xp0 = x
            self.yp0 = y
            self.wvp0 = wv
            
    def _chRaster(self):
        raster = _getRaster(self.data, self.wave, self.wv, self.dwv, hw=self.hw)
        M = raster.max()
        if M > 1e2:
            m = raster[raster > 1e2].min()
        else:
            m = raster.min()
        if self.scale == 'log':
            raster = np.log10(raster)
            m = np.log10(m)
            M = np.log10(M)
        self.imRaster.set_data(raster)
        self.vlineProfile.set_xdata(self.wv)
        self.vlineSpectro.set_xdata(self.wv)
        self.axProfile.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.band, self.wv))
        if self.scale == 'std':
            self.imRaster.set_clim(np.median(raster)-raster.std()*self.sigFactor, np.median(raster)+raster.std()*self.sigFactor)
        else:
            self.imRaster.set_clim(m, M)
        self.fig.canvas.draw_idle()

    def _chSpect(self):
        if self.cam == 'A':
            spectro = self.data[:, self.xp]
        elif self.cam == 'B':
            spectro = self.data[:, self.xp,::-1]
        if self.scale == 'log':
            spectro = np.log10(spectro)
        self.plotProfile.set_ydata(self.data[self.yp, self.xp])
        self.imSpectro.set_data(spectro)
        self.hlineSpectro.set_ydata(self.y)
        self.vlineRaster.set_xdata(self.x)
        self.pointRaster.set_offsets([self.x, self.y])

        ym = self.data[self.yp, self.xp].min()
        yM = self.data[self.yp, self.xp].max()
        margin = (yM-ym)*0.05
        self.axProfile.set_ylim(ym-margin, yM+margin)
        self.axSpectro.set_title(r"X = %.2f'', Y = %.2f'' (X$_{pix}$ = %i, Y$_{pix}$ = %i)"%(self.x, self.y, self.xp, self.yp))
        self.imSpectro.set_clim(spectro.min(), spectro.max())
        self.fig.canvas.draw_idle()

    def chRasterClim(self, cmin, cmax):
        self.imRaster.set_clim(cmin, cmax)
        self.fig.canvas.draw_idle()

    def chSpectroClim(self, cmin, cmax):
        self.imSpectro.set_clim(cmin, cmax)
        self.fig.canvas.draw_idle()

    def chcmap(self, cmap):
        self.imRaster.set_cmap(cmap)
        self.imSpectro.set_cmap(cmap)
        self.fig.canvas.draw_idle()

class dualBand:
    """
    Draw interactive FISS raster, spectrogram and profile for dual band.

    Parameters
    ----------
    fissA: `fisspy.read.FISS`
        FISS class.
    fissB: `fisspy.read.FISS`
        FISS class.
    x: `float`
        X position that you draw a spectral profile.
        Default is image center.
    y: `float`
        Y position that you draw a spectral profile.
        Default is image center.
    wvA: `float`
        Wavelength positin that you draw a raster images.
        Default is central wavelength.
    wvB: `float`
        Wavelength positin that you draw a raster images.
        Default is central wavelength.
    scale: `string`
        Scale method of colarbar limit.
        Default is minMax.
        option: 'minMax', 'std', 'log'
    sigFactor: `float`
        Factor of standard deviation.
        This is worked if scale is set to be 'std'
    helpBox: `bool`
        Show the interacitve key and simple explanation.
        Default is True

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.pyplot` properties
    """
    def __init__(self, fissA, fissB, x=None, y=None, wvA=None, wvB=None,
                 scale='log', sigFactor=3, helpBox=True, **kwargs):

        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        from ..align import alignOffset, shiftImage3D
        kwargs['interpolation'] = kwargs.pop('interpolation', 'bilinear')
        self.fissA = fissA
        self.fissB = fissB
        self.nx = self.fissA.nx
        self.nwvA = self.fissA.nwv
        self.nwvB = self.fissB.nwv
        self.dx = self.fissA.xDelt
        self.dy = self.fissA.yDelt
        self.dwvA = self.fissA.wvDelt
        self.dwvB = self.fissB.wvDelt
        if self.fissA.ny >= self.fissB.ny:
            self.dataA = self.fissA.data[:self.fissB.ny].copy()
            self.dataB = self.fissB.data.copy()
            self.ny = self.fissB.ny
            self.extentRaster = self.fissB.extentRaster
        elif fissA.ny < fissB.ny:
            self.dataB = self.fissB.data[:self.fissA.ny].copy()
            self.dataA = self.fissA.data.copy()
            self.ny = self.fissA.ny
            self.extentRaster = self.fissA.extentRaster
        self._xMin = self.extentRaster[0]
        self._xMax = self.extentRaster[1]
        self._yMin = self.extentRaster[2]
        self._yMax = self.extentRaster[3]

        sh = alignOffset(self.dataB[:,:,50], self.dataA[:,:,-50])
        tmp = shiftImage3D(fissB.data.transpose(2, 0, 1), -sh).transpose(1,2,0)
        self.dataB = tmp
        tmp[tmp<10]=1

        if not x:
            X = self.nx//2*self.dx
        else:
            X = x
        if not y:
            Y = self.ny//2*self.dy
        else:
            Y = y
        if not wvA:
            WVA = self.fissA.centralWavelength
        else:
            WVA = wvA
        if not wvB:
            WVB = self.fissB.centralWavelength
        else:
            WVB = wvB
        self.xp = xp = int(X/self.dx+0.5)
        self.yp = yp = int(Y/self.dy+0.5)
        self.x = xp*self.dx
        self.y = yp*self.dy
        self.scale = scale
        self.sigFactor = sigFactor
        self.hw = kwargs.pop('hw', 0.05)
        self.wvA = WVA
        self.wvB = WVB
        self.mwvA = self.fissA.wave[0]
        self.mwvB = self.fissB.wave[0]
        self.wvpA = int((WVA-self.mwvA)/self.dwvA+0.5)
        self.wvpB = int((WVB-self.mwvB)/self.dwvB+0.5)
        self.xp0 = self.xp
        self.yp0 = self.yp
        self.wvpA0 = self.wvpA
        self.wvpB0 = self.wvpB
        self.xpH = self.xp
        self.ypH = self.yp
        self.wvpAH = self.wvpA
        self.wvpBH = self.wvpB

        #Keyboard helpBox
        if helpBox:
            helpFig = plt.figure('Keyboard Help Box', figsize=[5,3])
            ax = helpFig.add_subplot(111)
            ax.set_position([0,0,1,1])
            ax.set_axis_off()
            ax.text(0.05,0.92,'ctrl+h: Reset to original setting')
            ax.text(0.05,0.82,'right: Move to right')
            ax.text(0.05,0.72,'left: Move to left')
            ax.text(0.05,0.62,'up: Move to up')
            ax.text(0.05,0.52,'down: Move to down')
            ax.text(0.05,0.42,'ctrl/cmd+right: Increase the wavelength of the fissA')
            ax.text(0.05,0.32,'ctrl/cmd+left: Decrease the wavelength of the fissA')
            ax.text(0.05,0.22,'ctrl/cmd+up: Increase the wavelength of the fissB')
            ax.text(0.05,0.12,'ctrl/cmd+down: Decrease the wavelength of the fissB')
            ax.text(0.05,0.02,'spacebar: Change to current mouse point')
            helpFig.show()

        #figure setting
        figsize = kwargs.pop('figsize', [12, 6])
        self.fig = plt.figure(figsize=figsize)
        # self.fig.canvas.set_window_title('Dual Band Image')
        self.imInterp = kwargs.get('interpolation', 'bilinear')
        gs = gridspec.GridSpec(2,4)
        self.axRasterA = self.fig.add_subplot(gs[:,0])
        self.axRasterB = self.fig.add_subplot(gs[:,1], sharex=self.axRasterA, sharey=self.axRasterA)
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
        ym = self.dataA[self.yp, self.xp].min()
        yM = self.dataA[self.yp, self.xp].max()
        margin = (yM-ym)*0.05
        self.axProfileA.set_ylim(ym-margin, yM+margin)
        ym = self.dataB[self.yp, self.xp].min()
        yM = self.dataB[self.yp, self.xp].max()
        margin = (yM-ym)*0.05
        self.axProfileB.set_ylim(ym-margin, yM+margin)
        self.axProfileA.minorticks_on()
        self.axProfileA.tick_params(which='both', direction='in')
        self.axProfileB.minorticks_on()
        self.axProfileB.tick_params(which='both', direction='in')

        #Draw
        rasterA = _getRaster(self.dataA, self.fissA.wave, self.wvA, self.dwvA, hw=self.hw)
        rasterB = _getRaster(self.dataB, self.fissB.wave, self.wvB, self.dwvB, hw=self.hw)

        MA = rasterA.max()
        if MA > 1e2:
            mA = rasterA[rasterA > 1e2].min()
        else:
            mA = rasterA.min()
        MB = rasterB.max()
        if MB > 1e2:
            mB = rasterB[rasterB > 1e2].min()
        else:
            mB = rasterB.min()

        if self.scale == 'log':
            rasterA = np.log10(rasterA)
            rasterB = np.log10(rasterB)
            MA = np.log10(MA)
            mA = np.log10(mA)
            MB = np.log10(MB)
            mB = np.log10(mB)
        
        self.imRasterA = self.axRasterA.imshow(rasterA, self.fissA.cmap, origin='lower', extent=self.extentRaster, **kwargs)
        self.imRasterB = self.axRasterB.imshow(rasterB, self.fissB.cmap, origin='lower', extent=self.extentRaster, **kwargs)
        self.plotProfileA = self.axProfileA.plot(self.fissA.wave, self.dataA[yp, xp], color='k')[0]
        self.plotProfileB = self.axProfileB.plot(self.fissB.wave, self.dataB[yp, xp], color='k')[0]

        if self.scale == 'std':
            self.imRasterA.set_clim(np.median(rasterA)-rasterA.std()*self.sigFactor, np.median(rasterA)+rasterA.std()*self.sigFactor)
            self.imRasterB.set_clim(np.median(rasterB)-rasterB.std()*self.sigFactor, np.median(rasterB)+rasterB.std()*self.sigFactor)
        else:
            self.imRasterA.set_clim(mA, MA)
            self.imRasterB.set_clim(mB, MB)

        #Reference
        self.vlineProfileA = self.axProfileA.axvline(self.wvA, ls='dashed', c='b')
        self.vlineProfileB = self.axProfileB.axvline(self.wvB, ls='dashed', c='b')
        self.pointRasterA = self.axRasterA.scatter(self.x, self.y, 50, marker='x', color='r')
        self.pointRasterB = self.axRasterB.scatter(self.x, self.y, 50, marker='x', color='r')
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)
        self.fig.show()

    def _onKey(self, event):

        if event.key == 'right':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.xp < self.nx-1:
                self.xp += 1
            else:
                self.xp = 0
            self.x = self.xp*self.dx
            self._chSpect()
        elif event.key == 'left':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.xp > 0:
                self.xp -= 1
            else:
                self.xp = self.nx-1
            self.x = self.xp*self.dx
            self._chSpect()
        elif event.key == 'up':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.yp < self.ny-1:
                self.yp += 1
            else:
                self.yp = 0
            self.y = self.yp*self.dy
            self._chSpect()
        elif event.key == 'down':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.yp > 0:
                self.yp -= 1
            else:
                self.yp = self.ny-1
            self.y = self.yp*self.dy
            self._chSpect()
        elif event.key == 'ctrl+right' or event.key == 'cmd+right':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.wvpA < self.nwvA-1:
                self.wvpA += 1
            else:
                self.wvpA = 0
            self.wvA = self.wvpA*self.dwvA+self.mwvA
            self._chRasterA()
        elif event.key == 'ctrl+left' or event.key == 'cmd+left':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.wvpA > 0:
                self.wvpA -= 1
            else:
                self.wvpA = self.nwvA-1
            self.wvA = self.wvpA*self.dwvA+self.mwvA
            self._chRasterA()
        elif event.key == 'ctrl+up' or event.key == 'cmd+up':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.wvpB < self.nwvB-1:
                self.wvpB += 1
            else:
                self.wvpB = 0
            self.wvB = self.wvpB*self.dwvB+self.mwvB
            self._chRasterB()
        elif event.key == 'ctrl+down' or event.key == 'cmd+down':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            if self.wvpB > 0:
                self.wvpB -= 1
            else:
                self.wvpB = self.nwvB-1
            self.wvB = self.wvpB*self.dwvB+self.mwvB
            self._chRasterB()
        elif event.key == ' ' and (event.inaxes == self.axRasterA or event.inaxes == self.axRasterB):
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            self.x = event.xdata
            self.y = event.ydata
            self.xp = int(self.x/self.dx+0.5)
            self.yp = int(self.y/self.dy+0.5)
            self.x = self.xp*self.dx
            self.y = self.yp*self.dy
            self._chSpect()
        elif event.key == ' ' and event.inaxes == self.axProfileA:
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            self.wvA = event.xdata
            self.wvpA = int((self.wvA-self.mwvA)/self.dwvA+0.5)
            self.wvA = self.wvpA*self.dwvA+self.mwvA
            self._chRasterA()
        elif event.key == ' ' and event.inaxes == self.axProfileB:
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            self.wvB = event.xdata
            self.wvpB = int((self.wvB-self.mwvB)/self.dwvB+0.5)
            self.wvB = self.wvpB*self.dwvB+self.mwvB
            self._chRasterB()
        elif event.key == 'ctrl+h' or event.key == 'cmd+h':
            self.xp0 = self.xp
            self.yp0 = self.yp
            self.wvpA0 = self.wvpA
            self.wvpB0 = self.wvpB
            self.wvpA = self.wvpAH
            self.wvpB = self.wvpBH
            self.xp = self.xpH
            self.yp = self.ypH
            self.x = self.xp*self.dx
            self.y = self.yp*self.dy
            self.wvA = self.wvpA*self.dwvA+self.mwvA
            self.wvB = self.wvpB*self.dwvB+self.mwvB
            self._chRasterA()
            self._chRasterB()
            self._chSpect()
        elif event.key == 'ctrl+b' or event.key == 'cmd+b':
            x = self.xp
            y = self.yp
            wvA = self.wvpA
            wvB = self.wvpB
            self.xp = self.xp0
            self.yp = self.yp0
            self.wvpA = self.wvpA0
            self.wvpB = self.wvpB0
            self.x = self.xp*self.dx
            self.y = self.yp*self.dy
            self.wvA = self.wvpA*self.dwvA+self.mwvA
            self.wvB = self.wvpB*self.dwvB+self.mwvB
            self._chRasterA()
            self._chRasterB()
            self._chSpect()
            self.xp0 = x
            self.yp0 = y
            self.wvpA0 = wvA
            self.wvpB0 = wvB

    def _chSpect(self):
        self.plotProfileA.set_ydata(self.dataA[self.yp, self.xp])
        self.plotProfileB.set_ydata(self.dataB[self.yp, self.xp])
        self.pointRasterA.set_offsets([self.x, self.y])
        self.pointRasterB.set_offsets([self.x, self.y])

        ym = self.dataA[self.yp, self.xp].min()
        yM = self.dataA[self.yp, self.xp].max()
        margin = (yM-ym)*0.05
        self.axProfileA.set_ylim(ym-margin, yM+margin)
        ym = self.dataB[self.yp, self.xp].min()
        yM = self.dataB[self.yp, self.xp].max()
        margin = (yM-ym)*0.05
        self.axProfileB.set_ylim(ym-margin, yM+margin)
        self.fig.canvas.draw_idle()

    def _chRasterA(self):
        rasterA = _getRaster(self.dataA, self.fissA.wave, self.wvA, self.fissA.wvDelt, hw=self.hw)
        M = rasterA.max()
        if M > 1e2:
            m = rasterA[rasterA > 1e2].min()
        else:
            m = rasterA.min()
        if self.scale == 'log':
            rasterA = np.log10(rasterA)
            m = np.log10(m)
            M = np.log10(M)
        self.imRasterA.set_data(rasterA)
        self.vlineProfileA.set_xdata(self.wvA)
        self.axProfileA.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.fissA.band, self.wvA))
        if self.scale == 'std':
            self.imRasterA.set_clim(np.median(rasterA)-rasterA.std()*self.sigFactor, np.median(rasterA)+rasterA.std()*self.sigFactor)
        else:
            self.imRasterA.set_clim(m, M)
        self.fig.canvas.draw_idle()

    def _chRasterB(self):
        rasterB = _getRaster(self.dataB, self.fissB.wave, self.wvB, self.fissB.wvDelt, hw=self.hw)
        M = rasterB.max()
        if M > 1e2:
            m = rasterB[rasterB > 1e2].min()
        else:
            m = rasterB.min()
        if self.scale == 'log':
            rasterB = np.log10(rasterB)
            m = np.log10(m)
            M = np.log10(M)
        self.imRasterB.set_data(rasterB)
        self.vlineProfileB.set_xdata(self.wvB)
        self.axProfileB.set_title(r'%s Band (wv = %.2f $\AA$)'%(self.fissB.band, self.wvB))
        if self.scale == 'std':
            self.imRasterB.set_clim(np.median(rasterB)-rasterB.std()*self.sigFactor, np.median(rasterB)+rasterB.std()*self.sigFactor)
        else:
            self.imRasterB.set_clim(m, M)
        self.fig.canvas.draw_idle()
