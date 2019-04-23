"""
"""

from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from os.path import join, dirname, basename
from scipy.fftpack import fft, fftfreq, ifft
from astropy.time import Time
import astropy.units as u
from glob import glob
from sunpy.cm import cm
import fisspy.cm as fisscm
from mpl_toolkits.axes_grid1 import ImageGrid

__author__= "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"


class FDView:
    
    def __init__(self, fname, frameRange=None, tavg=False):
        self.dirname = dirname(fname)
        self.basename = basename(fname)
        self.mask = fits.getdata(fname.replace('.','mask.'))
        self.time = fits.getdata(fname.replace('A.','t.'))
        self.dt = np.median(self.time-np.roll(self.time,1))*60
        self.FD = fits.getdata(fname)
        self.header = fits.getheader(fname)
        shape = self.FD.shape
        self.nt = shape[0]
        reftime = self.header['reftime']
        self.reftime = reftime
        isotRefTime = '%s-%s-%sT%s:%s:%s'%(reftime[:4],
                                        reftime[4:6],
                                        reftime[6:8],
                                        reftime[9:11],
                                        reftime[11:13],
                                        reftime[13:15])
        isotRefTime = Time(isotRefTime)
        self.isotime = isotRefTime + self.time *u.min
        self.t = 0
        self.rim = self.FD[:,:,:,0]
        self.dmap = np.nan_to_num(self.FD[:,:,:,2] * self.mask)
        
#        for n, dm in enumerate(self.dmap):
#            self.dmap[n] -= np.median(dm[dm != 0])
            
        self.dmap[self.mask == 0] = 0
        self.dmap0 = self.dmap.copy()
        
        if not frameRange:
            self.frameRange = [0, self.nt]
        else:
            self.set_frameRange(frameRange)
        if tavg:
            self.timeavg()
        
    def imshow(self, fnum=0, figsize=(10,6), dpi=100,
                 clim=[-3,3], cmap=plt.cm.RdBu_r, level=75):
        self.t = fnum
        self.lev0 = level
        self.fig, self.ax = plt.subplots(1,2, figsize=figsize)
        self.raster = self.ax[0].imshow(self.rim[self.t], cmap=plt.cm.gray,
                             origin='lower',
                             interpolation='bilinear')
        self.doppler = self.ax[1].imshow(self.dmap[self.t], cmap=cmap, 
                             origin='lower', clim=clim,
                             interpolation='bilinear')
        self.con = self.ax[1].contour(self.rim[self.t], colors='k',
                          origin='lower',
                          levels=[self.rim[self.t].max()/100*self.lev0])
        self.fig.suptitle('%s    fnum : %i'%(self.isotime[self.t].value, self.t))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.tight_layout(rect=[0,0,1,0.95])
        self.ax[0].set_title('%s'%self.header['ID0'])
        self.ax[1].set_title('%s'%self.header['ID2'])
        
    def FourierFilter(self, filterRange):
        
        self.freq = fftfreq(self.nt, self.dt)*1e3
        filt = np.logical_or(np.abs(self.freq) <= filterRange[0],
                             np.abs(self.freq) >= filterRange[1])
        
        ftd = fft(self.dmap, axis=0)
        ftd[filt] = 0
        self.fdmap = ifft(ftd, axis=0).real
        self.fdmap[self.mask == 0] = 0
        self.dmap = self.fdmap.copy()
        

    def _on_key(self, event):
        if event.key == 'right':
            if self.t < self.nt-1:
                self.t +=1
            else:
                self.t = 0
            self.raster.set_data(self.rim[self.t])
            self.doppler.set_data(self.dmap[self.t])
            self.con.collections[0].remove()
            self.con = self.ax[1].contour(self.rim[self.t], colors='k',
                  origin='lower',
                  levels=[self.rim[self.t].max()/100*self.lev0])
            
        elif event.key == 'left':
            if self.t > 0:
                self.t -=1
            else:
                self.t = self.nt-1
            self.raster.set_data(self.rim[self.t])
            self.doppler.set_data(self.dmap[self.t])
            self.con.collections[0].remove()
            self.con = self.ax[1].contour(self.rim[self.t], colors='k',
                  origin='lower',
                  levels=[self.rim[self.t].max()/100*self.lev0])
        self.fig.suptitle('%s    fnum : %i'%(self.isotime[self.t].value, self.t))
        self.fig.canvas.draw_idle()
    
    def chclim(self, v_clim, raster_clim=False):
        self.doppler.set_clim(v_clim)
        if raster_clim:
            self.raster(raster_clim)
            
    def setSection(self, xlim, ylim):
        self.ax[0].set_xlim(xlim)
        self.ax[0].set_ylim(ylim)
        self.ax[1].set_ylim(ylim)
        self.ax[1].set_xlim(xlim)
        
    def set_interpolation(self, intp):
        self.raster.set_interpolation(intp)
        self.doppler.set_interpolation(intp)
        
    def timeavg(self):
        self.dmap -= np.median(self.dmap, 0)
    
    def odata(self):
        self.data = self.data0.copy()
        self.nt = self.data.shape[0]
        self.frameRange = [0, self.nt]
        
    def set_frameRange(self, frameRange):
        if frameRange[1] == -1:
            frameRange[1] = self.nt
        self.dmap = self.dmap[frameRange[0]:frameRange[1]]
        self.rim = self.rim[frameRange[0]:frameRange[1]]
        self.mask = self.mask[frameRange[0]:frameRange[1]]
        self.isotime = self.isotime[frameRange[0]:frameRange[1]]
        self.nt = frameRange[1]-frameRange[0]
        self.frameRange = frameRange
        print('Frame range: %s'%self.frameRange)
        
        
    

class TiOView:
    
    def __init__(self, dirn, xlim=None, ylim=None):
        self.flist = glob(join(dirn,'*.fts'))
        self.flist.sort()
        self.TiO = []
        self.header = []
        self.obstime = []
        for n, f in enumerate(self.flist):
            try:
                self.TiO += [fits.getdata(f)]
                if xlim and ylim:
                    self.TiO[-1] = self.TiO[-1][ylim[0]:ylim[1],xlim[0]:xlim[1]]
                self.header += [fits.getheader(f)]
                self.obstime += [self.header[-1]['time-obs']]
            except:
                pass
        self.nt = len(self.TiO)
    def imshow(self, fnum=0, figsize=(8,8), dpi=100, cmap=plt.cm.gray):
        self.t = 0
        self.fig, self.ax =plt.subplots(figsize=figsize)
        self.im = self.ax.imshow(self.TiO[self.t], cmap, origin='lower')
        self.ax.set_title(self.obstime[self.t])
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _on_key(self, event):
        if event.key == 'right':
            if self.t < self.nt-1:
                self.t +=1
            else:
                self.t = 0
            
        elif event.key == 'left':
            if self.t > 0:
                self.t -=1
            else:
                self.t = self.nt-1
        self.im.set_data(self.TiO[self.t])
        self.ax.set_title(self.obstime[self.t])
        self.fig.canvas.draw_idle()
        
class FISS_TiO_View:
    def __init__(self, FDfile, dirTiO, fscale=0.16, tscale= 0.034,
                 xoff=0, yoff=0):
        
        flist = glob(join(dirTiO,'*.fts'))
        self.TiOlist = flist.sort()

        headerTiO = []
        self.isotTiO = []
        self.TiO = []
        for i in flist:
            self.TiO += [fits.getdata(i)]
            headerTiO += [fits.getheader(i)]
            obstime = headerTiO[-1]['time-obs']
            obsdate = headerTiO[-1]['date-obs']
            self.isotTiO += ['%sT%s'%(obsdate, obstime)]
        self.isotTiO = Time(self.isotTiO)
        tny, tnx = self.TiO[0].shape
        tyl = tny*tscale/2
        txl = tnx*tscale/2
        
        FD = fits.getdata(FDfile)
        self.mask = fits.getdata(FDfile.replace('.','mask.'))
        FDtime = fits.getdata(FDfile.replace('A.', 't.'))
        self.dt = np.median(FDtime - np.roll(FDtime, 1))*60
        self.rim = FD[:,:,:,0]
        self.dmap = np.nan_to_num(FD[:,:,:,2]*self.mask)
        self.dmapp = np.nan_to_num(FD[:,:,:,4]*self.mask)
        
        for n, dm in enumerate(self.dmap):
            self.dmap[n] -= np.median(dm[dm != 0])
            self.dmapp[n] -= np.median(self.dmapp[n][self.dmapp[n] != 0])
        self.dmapp -= np.median(self.dmapp, 0)
        self.dmap -= np.median(self.dmap, 0)
        self.fnt, fny, fnx = self.dmap.shape
        fxl = fnx*fscale/2
        fyl = fny*fscale/2
        

        headerFD = fits.getheader(FDfile)
        reftime = headerFD['reftime']
        isotRefTime = '%s-%s-%sT%s:%s:%s'%(reftime[:4],
                                            reftime[4:6],
                                            reftime[6:8],
                                            reftime[9:11],
                                            reftime[11:13],
                                            reftime[13:15])
        isotRefTime = Time(isotRefTime)
        self.isoFDtime = isotRefTime + FDtime*u.min
        
        
        
        
        self.t=0
#        xoff = -0.8
#        yoff = +1.7
        self.fextent = [-fxl, fxl, -fyl, fyl]
        self.textent = [-txl+xoff, txl+xoff, -tyl+yoff, tyl+yoff]
        
        
        
    def imshow(self, fnum=0, figsize=(15,8), dpi=100,
               xlim=None, ylim=None, clim=[-3,3],
               climp=[-0.3,0.3],
               cmap=plt.cm.RdBu_r,
               lev0=75):
        self.t = fnum
        self.lev0 = lev0
        dt = self.isotTiO - self.isoFDtime[self.t]
        wh = np.abs(dt.value).argmin()
        self.fig, self.ax = plt.subplots(1,3, figsize=figsize, dpi=dpi)
        
        self.Dimp = self.ax[0].imshow(self.dmapp[self.t], plt.cm.RdBu_r,
                          origin='lower',extent=self.fextent,
                          clim=climp)
        self.Dim = self.ax[1].imshow(self.dmap[self.t], plt.cm.RdBu_r,
                          origin='lower',extent=self.fextent,
                          clim=clim)
        self.Tim = self.ax[2].imshow(self.TiO[wh], plt.cm.gray,
                          origin='lower', extent=self.textent)
        self.con0 = self.ax[0].contour(self.rim[self.t], colors='k',
                           origin='lower',
                           levels=[self.rim[self.t].max()*75/100],
                           extent=self.fextent)
        self.con1 = self.ax[1].contour(self.rim[self.t], colors='k',
                           origin='lower',
                           levels=[self.rim[self.t].max()*75/100],
                           extent=self.fextent)
        self.con2 = self.ax[2].contour(self.rim[self.t],
                           colors='lime', origin='lower',
                           levels=[self.rim[self.t].max()*lev0/100],
                           extent=self.fextent)
                
#        xlim = [-7,8]
#        ylim = [-10,5]
        self.ax[0].set_xlim(xlim)
        self.ax[1].set_xlim(xlim)
        self.ax[2].set_xlim(xlim)
        self.ax[0].set_ylim(ylim)
        self.ax[1].set_ylim(ylim)
        self.ax[2].set_ylim(ylim)
        self.ax[0].set_title(self.isoFDtime[self.t])
        self.ax[1].set_title(self.isoFDtime[self.t])
        self.ax[2].set_title(self.isotTiO[wh])
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def scatter(self, sx, sy, size=30**2, edgecolors='r',
                linewidths=2):
#        sx = [1.94]
#        sy = [-2.77]
        self.sc0 = self.ax[0].scatter(sx, sy, size,
                          edgecolors=edgecolors, facecolors="None",
                          linewidths=linewidths)
        self.sc1 = self.ax[1].scatter(sx, sy, size,
                          edgecolors=edgecolors, facecolors="None",
                          linewidths=linewidths)
        self.sc2 = self.ax[2].scatter(sx, sy, size,
                          edgecolors=edgecolors, facecolors="None",
                          linewidths=linewidths)

    def chscatter(self, sx, sy, size=30**2, edgecolors='r',
                  linewidths=2):
        self.sc0.remove()
        self.sc1.remove()
        self.sc2.remove()
        self.scatter(sx, sy, size=size, edgecolors='r',
                     linewidths=linewidths)
    
    def _on_key(self, event):
        if event.key == 'right':
            if self.t < self.fnt-1:
                self.t +=1
            else:
                self.t = 0

        elif event.key == 'left':
            if self.t > 0:
                self.t -=1
            else:
                self.t = self.fnt-1
                
        dt = self.isotTiO - self.isoFDtime[self.t]
        wh = np.abs(dt.value).argmin()
        self.ax[0].set_title(self.isoFDtime[self.t])
        self.ax[1].set_title(self.isoFDtime[self.t])
        self.ax[2].set_title(self.isotTiO[wh])
        self.Dimp.set_data(self.dmapp[self.t])
        self.Dim.set_data(self.dmap[self.t])
        self.Tim.set_data(self.TiO[wh])
        self.con0.collections[0].remove()
        self.con1.collections[0].remove()
        self.con2.collections[0].remove()
        self.con0 = self.ax[0].contour(self.rim[self.t],
                   colors='k', origin='lower',
                   levels=[self.rim[self.t].max()*self.lev0/100],
                   extent=self.fextent)
        self.con1 = self.ax[1].contour(self.rim[self.t],
                   colors='k', origin='lower',
                   levels=[self.rim[self.t].max()*self.lev0/100],
                   extent=self.fextent)
        self.con2 = self.ax[2].contour(self.rim[self.t],
                   colors='lime', origin='lower',
                   levels=[self.rim[self.t].max()*self.lev0/100],
                   extent=self.fextent)
        
        self.fig.canvas.draw_idle()
        
        
        
    def FourierFilter(self, frameRange=None, filterRange=None):
        if not frameRange:
            frameRange = [0, self.fnt]
        if not filterRange:
            KeyError('filterRange must be given')
        
        print(frameRange)
        fnt = frameRange[1]-frameRange[0]
        self.fnt = fnt
        self.rim = self.rim[frameRange[0]:frameRange[1]]
        data = self.dmap[frameRange[0]:frameRange[1]]
        datap = self.dmapp[frameRange[0]:frameRange[1]]
        self.freq = fftfreq(fnt, self.dt)*1e3
        filt = np.logical_or(np.abs(self.freq) < filterRange[0],
                             np.abs(self.freq) > filterRange[1])
        
        ftd = fft(data, axis=0)
        ftdp = fft(datap, axis=0)
        ftd[filt] = 0
        ftdp[filt] = 0
        self.fdmap = ifft(ftd, axis=0).real
        self.fdmapp = ifft(ftdp, axis=0).real
        self.fdmap[self.mask[frameRange[0]:frameRange[1]] == 0] = 0
        self.fdmapp[self.mask[frameRange[0]:frameRange[1]] == 0] = 0
        self.dmap = self.fdmap.copy()
        self.dmapp = self.fdmapp.copy()
        
    def setSection(self, xlim, ylim):
        self.ax[0].set_xlim(xlim)
        self.ax[1].set_xlim(xlim)
        self.ax[2].set_xlim(xlim)
        self.ax[0].set_ylim(ylim)
        self.ax[1].set_ylim(ylim)
        self.ax[2].set_ylim(ylim)
        
        
class IRISView:
    def __init__(self, fname):
        self.filename = fname
        self.basename = basename(fname)
        self.dirname = dirname(fname)
        self.header = fits.getheader(fname)
        self.startTime = Time(self.header['startobs'])
        self.endTime = Time(self.header['endobs'])
        self.instrument = r'%s/%s %i $\AA$'%(self.header['telescop'],
                                             self.header['instrume'],
                                             self.header['twave1'])
        self.dt = self.header['cdelt3']
        self.nt = self.header['naxis3']
        self.wavelen = int(self.header['twave1'])
        self.cmap = cm.get_cmap('irissji%i'%self.wavelen)
        self.tarr = self.startTime + \
                    np.arange(self.nt)*self.dt*u.second
                    
        self.data0 = fits.getdata(fname)
        self.data = self.data0.copy()
        
        self.fnum = 0
        nx = self.header['naxis1']
        ny = self.header['naxis2']
        dx = self.header['cdelt1']
        dy = self.header['cdelt2']
        xc = self.header['crpix1']
        yc = self.header['crpix2']
        rx = self.header['crval1']
        ry = self.header['crval2']
        self.extent = [rx+(xc-(nx-1))*dx,
                       rx+(xc+(nx-1)/2)*dx,
                       ry+(yc-(ny-1))*dy,
                       ry+(yc+(ny-1)/2)*dy]
        
    def imshow(self, fnum=0, figsize=(8,8), dpi=100, clim=[30,700]):
        self.fnum = fnum
        
        self.fig, self.ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
        self.im = self.ax.imshow(self.data[self.fnum],
                                 self.cmap,
                                 origin='lower',
                                 extent=self.extent,
                                 clim=clim,
                                 interpolation='bilinear')
        self.ax.set_title('%s   %s - (%i/%i)'%(self.instrument,
                                           self.tarr[self.fnum].value,
                                           self.fnum,
                                           self.nt))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _on_key(self, event):
        if event.key == 'right':
            if self.fnum < self.nt-1:
                self.fnum += 1
            else:
                self.fnum = 0

        elif event.key == 'left':
            if self.fnum > 0:
                self.fnum -=1
            else:
                self.fnum = self.nt-1
        
        self.ax.set_title('%s   %s - (%i/%i)'%(self.instrument,
                                           self.tarr[self.fnum].value,
                                           self.fnum,
                                           self.nt))
        self.im.set_data(self.data[self.fnum])
        self.fig.canvas.draw_idle()
        
    def mdata(self):
        self.data = self.data0 - np.median(self.data0, 0)
        
    def odata(self):
        self.data = self.data0.copy()
        
    def chclim(self, clim):
        self.im.set_clim(clim)
    
    def chcmap(self, cmap):
        self.im.set_cmap(cmap)
        
    def subSection(self, xlim, ylim):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
    def FourierFilter(self, filterRange=None):
        if not filterRange:
            KeyError('filterRange must be given')
        
        self.freq = fftfreq(self.nt, self.dt)*1e3
        filt = np.logical_or(np.abs(self.freq) <= filterRange[0],
                             np.abs(self.freq) >= filterRange[1])
        
        ftd = fft(self.data, axis=0)
        ftd[filt] = 0
        self.fdata = ifft(ftd, axis=0).real
        self.data = self.fdata.copy()
        
class AIACubeView:
    
    def __init__(self, fname):
        self.filename = fname
        self.basename = basename(fname)
        self.dirname = dirname(fname)
        self.header = fits.getheader(fname)
        self.startTime = Time(self.header['startobs'])
        self.endTime = Time(self.header['endobs'])
        self.wavelen = self.basename.split('_')[1]
        self.instrument = r'SDO/AIA %s $\AA$'%self.wavelen
        self.cmap = cm.get_cmap('sdoaia%s'%self.wavelen)
        if self.wavelen == '1600':
            self.dt = 24
        else:
            self.dt = self.header['cdelt3']
        self.nt = self.header['naxis3']
        self.tarr = self.startTime +\
                    np.arange(self.nt)*self.dt*u.second
        self.data0 = fits.getdata(fname)
        self.data = self.data0.copy()
        
        nx = self.header['naxis1']
        ny = self.header['naxis2']
        dx = self.header['cdelt1']
        dy = self.header['cdelt2']
        xc = self.header['crpix1']
        yc = self.header['crpix2']
        rx = self.header['crval1']
        ry = self.header['crval2']
        self.extent = [rx-xc*dx,
                       rx+(nx-xc)*dx,
                       ry-yc*dy,
                       ry+(ny-yc)*dy]
        
    def imshow(self, fnum=0, figsize=(8,8), dpi=100, clim=[30,700]):
        self.fnum = fnum
        
        self.fig, self.ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
        self.im = self.ax.imshow(self.data[self.fnum],
                                 self.cmap,
                                 origin='lower',
                                 extent=self.extent,
                                 clim=clim,
                                 interpolation='bilinear')
        self.ax.set_title('%s   %s - (%i/%i)'%(self.instrument,
                                           self.tarr[self.fnum].value,
                                           self.fnum,
                                           self.nt))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _on_key(self, event):
        if event.key == 'right':
            if self.fnum < self.nt-1:
                self.fnum += 1
            else:
                self.fnum = 0

        elif event.key == 'left':
            if self.fnum > 0:
                self.fnum -=1
            else:
                self.fnum = self.nt-1
        
        self.ax.set_title('%s   %s - (%i/%i)'%(self.instrument,
                                           self.tarr[self.fnum].value,
                                           self.fnum,
                                           self.nt))
        self.im.set_data(self.data[self.fnum])
        self.fig.canvas.draw_idle()
        
    def mdata(self):
        self.data = self.data0 - np.median(self.data0, 0)
        
    def odata(self):
        self.data = self.data0.copy()
        
    def chclim(self, clim):
        self.im.set_clim(clim)
    
    def chcmap(self, cmap):
        self.cmap = cmap
        self.im.set_cmap(cmap)
        
    def cmapReverse(self):
        self.cmap = self.cmap.reversed()
        self.im.set_cmap(self.cmap)
        
    def subSection(self, xlim, ylim):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
    def FourierFilter(self, filterRange):
        self.freq = fftfreq(self.nt, self.dt)*1e3
        filt = np.logical_or(np.abs(self.freq) <= filterRange[0],
                             np.abs(self.freq) >= filterRange[1])
        
        ftd = fft(self.data, axis=0)
        ftd[filt] = 0
        self.fdata = ifft(ftd, axis=0).real
        self.data = self.fdata.copy()
        
class AIAmultiCube:
    
    def __init__(self, flist, reftimeFilter='304', fdname=None, fdinum=[1]):
        if fdname:
            self.fdcmap = [fisscm.ha, fisscm.ha, plt.cm.RdBu_r,
                           fisscm.ha, plt.cm.RdBu_r]
            fdinum = np.array(fdinum).flatten()
        nfdinum = len(fdinum)
        self.flist = flist
        self.nflist = len(flist) + nfdinum
        self.reftimeFilter = reftimeFilter
        self.basename = [None]*self.nflist
        self.dirname = [None]*self.nflist
        self.header = [None]*self.nflist
        self.startTime = [None]*self.nflist
        self.endTime = [None]*self.nflist
        self.wavelen = [None]*self.nflist
        self.instrument = [None]*self.nflist
        self.cmap = [None]*self.nflist
        self.dt = [None]*self.nflist
        self.nt = [None]*self.nflist
        self.tarr = [None]*self.nflist
        self.data0 = [None]*self.nflist
        self.data = [None]*self.nflist
        self.extent = [None]*self.nflist
        jd = [None]*self.nflist
        self.dtjd = [None]*self.nflist
        self.armin = [None]*self.nflist
        

        if fdname:
            fdcmap = [fisscm.ha, fisscm.ha, plt.cm.RdBu_r,
                      fisscm.ha, plt.cm.RdBu_r]
            self.fdheader = fits.getheader(fdname)
            nx = self.fdheader['naxis2']
            ny = self.fdheader['naxis3']
            xpos = self.fdheader['xpos']
            ypos = self.fdheader['ypos']
            fdextent = [xpos-0.16*(nx-1)/2, xpos+0.16*(nx-1)/2,
                        ypos-0.16*(ny-1)/2, ypos+0.16*(ny-1)/2]
            time = fits.getdata(fdname.replace('A','t'))
            mask = fits.getdata(fdname.replace('.','mask.'))
            
            reftime = self.fdheader['reftime']
            isotRefTime = '%s-%s-%sT%s:%s:%s'%(reftime[:4],
                                        reftime[4:6],
                                        reftime[6:8],
                                        reftime[9:11],
                                        reftime[11:13],
                                        reftime[13:15])
            isotRefTime = Time(isotRefTime)
            fddata_ori = fits.getdata(fdname)
            fdbase = basename(fdname)
            fddir = dirname(fdname)
            for n, i in enumerate(fdinum):
                self.basename[n] = fdbase
                self.dirname[n] = fddir
                self.header[n] = self.fdheader
                self.dt[n] = np.median(time-np.roll(time,1))*60
                self.tarr[n] = isotRefTime + time*u.min
                self.startTime[n] = self.tarr[n][0]
                self.endTime[n] = self.tarr[n][-1]
                self.wavelen[n] = self.fdheader['ID%i'%i]
                self.instrument[n] = 'GST / FISS'
                self.cmap[n] = fdcmap[i]
                self.nt[n] = fddata_ori.shape[0]
                self.extent[n] = fdextent
                data0 = np.nan_to_num(fddata_ori[:,:,:,i]*mask)
                for t, dm in enumerate(data0):
                    data0[t] -= np.median(dm[dm != 0])
                data0[mask == 0] = 0
                self.data0[n] = data0
                self.data[n] = data0.copy()
                
            
        for i,f in enumerate(flist):
            n = i+nfdinum
            self.basename[n] = basename(f)
            self.dirname[n] = dirname(f)
            self.header[n] = fits.getheader(f)
            self.startTime[n] = Time(self.header[n]['startobs'])
            self.endTime[n] = Time(self.header[n]['endobs'])
            self.wavelen[n] = self.basename[n].split('_')[1]
            self.instrument[n] = r'SDO/AIA %s $\AA$'%self.wavelen[n]
            self.cmap[n] = cm.get_cmap('sdoaia%s'%self.wavelen[n])
            self.dt[n] = self.header[n]['cdelt3']
            self.nt[n] = self.header[n]['naxis3']
            self.tarr[n] = self.startTime[n] +\
                            np.arange(self.nt[n])*self.dt[n]*u.second
            self.data0[n] = fits.getdata(f)
            self.data = self.data0.copy()
        
            nx = self.header[n]['naxis1']
            ny = self.header[n]['naxis2']
            dx = self.header[n]['cdelt1']
            dy = self.header[n]['cdelt2']
            xc = self.header[n]['crpix1']
            yc = self.header[n]['crpix2']
            rx = self.header[n]['crval1']
            ry = self.header[n]['crval2']
            self.extent[n] = [rx-xc*dx,
                           rx+(nx-xc)*dx,
                           ry-yc*dy,
                           ry+(ny-yc)*dy]
        
        

            
            
        # time alignment
        self.refIndex = np.where(np.array(self.wavelen) == reftimeFilter)[0][0]
        self.refnt = self.nt[self.refIndex]
        self.xc = self.header[self.refIndex]['crval1']
        self.yc = self.header[self.refIndex]['crval2']
        for i in range(self.nflist):
            jd = self.tarr[i].jd
            self.dtjd[i] = np.abs(jd[:,None]-self.tarr[self.refIndex].jd)
            self.armin[i] = self.dtjd[i].argmin(0)
        
        
    def imshow(self, nrows, ncols, fnum=0, figsize=(12,8), dpi=100,
               interpolation='bilinear'):
        
        self.fnum = fnum
        self.fig = plt.figure(figsize=figsize,dpi=dpi)
        self.grid = ImageGrid(self.fig, 111,
                              nrows_ncols=(nrows, ncols),
                              axes_pad=0.,
                              share_all=True,
                              label_mode='1')
        self.im = [None]*self.nflist
        self.txt = [None]*self.nflist
        
        for i in range(self.nflist):
            self.im[i] = self.grid[i].imshow(self.data[i][self.armin[i][self.fnum]],
                                             self.cmap[i],
                                             origin='lower',
                                             extent=self.extent[i],
                                             interpolation=interpolation)
            time = self.tarr[i][self.armin[i][self.fnum]].value.split('T')[1].split('.')[0]
            self.txt[i] = self.grid[i].text(0.05, 0.9,
                              r'%s $\AA$   %s UT'%(self.wavelen[i], time),
                              bbox=dict(boxstyle="round", ec='none',
                                        fc='w', alpha=0.5),
                              transform=self.grid.axes_all[i].transAxes,
                              fontsize=12,
                              fontweight='bold')
            self.grid.axes_all[i].tick_params(direction='in')
        self.fig.suptitle('%s - (%i/%i)'%(self.tarr[0][0].value.split('T')[0],
                                          self.fnum, self.refnt),
                          fontsize=15, fontweight='bold')
        self.grid.axes_llc.tick_params(labelsize=12)
        self.grid.axes_llc.set_xlabel('X (arcsec)', fontsize=12)
        self.grid.axes_llc.set_ylabel('Y (arcsec)', fontsize=12)
        self.fig.tight_layout(rect=[0,0,1,0.97])
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _on_key(self, event):
        if event.key == 'right':
            if self.fnum < self.refnt-1:
                self.fnum += 1
            else:
                self.fnum = 0

        elif event.key == 'left':
            if self.fnum > 0:
                self.fnum -=1
            else:
                self.fnum = self.refnt-1
        
        for i in range(self.nflist):
            self.txt[i].remove()
            time = self.tarr[i][self.armin[i][self.fnum]].value.split('T')[1].split('.')[0]
            self.txt[i] = self.grid[i].text(0.05, 0.9,
                              r'%s $\AA$   %s UT'%(self.wavelen[i], time),
                              bbox=dict(boxstyle="round", ec='none',
                                        fc='w', alpha=0.5),
                              transform=self.grid.axes_all[i].transAxes,
                              fontsize=12,
                              fontweight='bold')
            self.im[i].set_data(self.data[i][self.armin[i][self.fnum]])
        self.fig.suptitle('%s - (%i/%i)'%(self.tarr[0][0].value.split('T')[0],
                                          self.fnum, self.refnt),
                          fontsize=15, fontweight='bold')
        self.fig.canvas.draw_idle()
        
    def mdata(self):
        for i in range(self.nflist):
            self.data[i] = self.data0[i] - np.median(self.data0[i], 0)
            
    def odata(self):
        for i in range(self.nflist):
            self.data[i] = self.data0[i].copy()
        
    def chclim(self, climlist):
        if len(climlist) != self.nflist:
            raise ValueError('The number of climlist should be same with the number of flist')
        for n,clim in enumerate(climlist):
            self.im[n].set_clim(clim)
            
    def FourierFilter(self, filterRange):
        for i in range(self.nflist):
            freq = fftfreq(self.nt[i], self.dt[i])*1e3
            filt = np.logical_or(np.abs(freq) <= filterRange[0],
                                 np.abs(freq) >= filterRange[1])
            ftd = fft(self.data[i], axis=0)
            ftd[filt] = 0
            self.data[i] = ifft(ftd, axis=0).real
            
    def subSection(self, xlim, ylim):
        self.grid.axes_llc.set_xlim(xlim)
        self.grid.axes_llc.set_ylim(ylim)
        
    def set_interpolation(self, interpolation):
        for i in range(self.nflist):
            self.im[i].set_interpolation(interpolation)