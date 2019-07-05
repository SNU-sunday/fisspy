from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fisspy.analysis.filter import FourierFilter
from interpolation.splines import LinearSpline
from matplotlib.animation import FuncAnimation
import astropy.units as u
from astropy.time import Time

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

class TDmap:
    """
    Make Time-Distance map for given slit position
    
    Parameters
    ----------
    data : `~numpy.ndarray`
        3-dimensional data array (time, y, x).
    header : '~astropy.io.fits.header.Header
        Header of data.
    tarr : `~numpy.ndarray`, optional
        Array of time (unit: second).
    filterRange : `list`, optional
        List of range of Fourier bandpass filters
        
    Returns
    -------
    td : `~fisspy.analysis.tdmap.TDmap`
        A new time distance class object.
    
    Examples
    --------
    
    """
    
    def __init__(self, data, header, tarr=None, filterRange=None, cmap=None):
        
        self.data = data
        self.header = header
        self.nx = self.header['naxis1']
        self.ny = self.header['naxis2']
        self.nt = self.header['naxis3']
        self.dx = self.header['cdelt1']
        self.dy = self.header['cdelt2']
        self.dt = self.header['cdelt3']
        self.rx = self.header['crval1']
        self.ry = self.header['crval2']
        self.cmap = cmap
            
        if not np.any(tarr):
            tarr = np.arange(0, self.nt*self.dt, self.dt)
        self._tarr = tarr
        self.Time = Time(self.header['sttime']) + tarr*u.second
        
        self.extent = [self.rx-self.nx/2*self.dx,
                       self.rx+self.nx/2*self.dx,
                       self.ry-self.ny/2*self.dy,
                       self.ry+self.ny/2*self.dy]
        self._xarr = np.linspace(self.extent[0]+self.dx*0.5,
                                 self.extent[1]-self.dx*0.5,
                                 self.nx)
        self._yarr = np.linspace(self.extent[2]+self.dy*0.5,
                                 self.extent[3]-self.dy*0.5,
                                 self.ny)
        
        self.smin = [self._tarr[0],
                     self.extent[2]+0.5*self.dy,
                     self.extent[0]+0.5*self.dx]
        self.smax = [self._tarr[-1],
                     self.extent[3]-0.5*self.dy,
                     self.extent[1]-0.5*self.dx]
        self.order = [self.nt, self.ny, self.nx]
        
        self._tname = ['ori']
        if not filterRange:
            self.nfilter = 1
            self.fdata = np.empty([1, self.nt, self.ny, self.nx])
        else:
            self.nfilter = len(filterRange)+1
            self.fdata = np.empty([self.nfilter, self.nt, self.ny, self.nx])
            
            for n, fR in enumerate(filterRange):
                self._tname += ['%.1f - %.1f mHZ'%(fR[0], fR[1])]
                self.fdata[n+1] = FourierFilter(self.data, self.nt,
                          self.dt*1e-3, fR)
            
        self.fdata[0] = self.data
        self.interp = []
        for data in self.fdata:
            self.interp += [LinearSpline(self.smin, self.smax,
                                         self.order, data)]
        
    def get_TD(self, R, xc, yc, angle):
        self.R = R
        self.xc = xc
        self.yc = yc
        self.angle = angle
        ang = np.deg2rad(self.angle)
        nl = int(np.ceil(2*R/self.dx))
        self.x1 = -R*np.cos(ang) + xc
        self.x2 = R*np.cos(ang) + xc
        self.y1 = -R*np.sin(ang) + yc
        self.y2 = R*np.sin(ang) + yc
        x = np.linspace(self.x1, self.x2, nl)
        y = np.linspace(self.y1, self.y2, nl)
        
        oiarr = np.empty([nl, self.nt, 3])
        oiarr[:,:,0] = self._tarr
        oiarr[:,:,1] = y[:,None]
        oiarr[:,:,2] = x[:,None]
        iarr = oiarr.reshape([nl*self.nt, 3])
        
        td = self.interp[self.filterNum-1](iarr)
        
        return td.reshape([nl, self.nt])
    
    def imshow(self, R=5, xc=None, yc=None, angle=0, t=0,
               filterNum=1, fps=10, cmap=plt.cm.gray,
               interpolation='bilinear'):
        
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        if not xc:
            xc = self.rx
        if not yc:
            yc = self.ry
        self.R = self._R0 = R
        self.angle = self._angle0 = angle
        self.xc = self._xc0 = xc
        self.yc = self._yc0 = yc
        self.filterNum = self._filterNum0 = filterNum
        self.t = self._t0 = t
        self.fps = fps
        self.pause = 'ini'
        self.pos = []
        self.mark = []
        self.hlines = []
        tpix = np.abs(self._tarr-self.t).argmin()
        self.td = self.get_TD(R,xc,yc,angle)
        self.tdextent = [self._tarr[0]-0.5*self.dt,
                         self._tarr[-1]+0.5*self.dt,
                         -self.R,
                         self.R]
        if not self.cmap:
            self.cmap = cmap
        
        self.fig= plt.figure(figsize=[14,9])
        self.fig.canvas.set_window_title('%s ~ %s'%(self.Time[0], self.Time[-1]))
        gs = gridspec.GridSpec(5, self.nfilter)
        
        self.axTD = self.fig.add_subplot(gs[3:, :])
        self.axTD.set_xlabel('Time (sec)')
        self.axTD.set_ylabel('Distance (arcsec)')
        self.axTD.set_title('%i: %s,  '
                            'Time: %s, '
                            'tpix: %i'%(filterNum, self._tname[filterNum-1],
                                        self.Time[tpix].value,
                                        tpix))
        self.imTD = self.axTD.imshow(self.td,
                                     extent=self.tdextent,
                                     origin='lower',
                                     cmap=self.cmap,
                                     interpolation=interpolation)
        
        self.axRaster = []
        self.im = []
        for i in range(self.nfilter):
            if i == 0:
                self.axRaster += [self.fig.add_subplot(gs[:3, i])]
                self.axRaster[i].set_xlabel('X (arcsec)')
                self.axRaster[i].set_ylabel('Y (arcsec)')
            else:
                self.axRaster += [self.fig.add_subplot(gs[:3, i],
                                                       sharex=self.axRaster[0],
                                                       sharey=self.axRaster[0])]
                self.axRaster[i].tick_params(labelleft=False, labelbottom=False)
            self.axRaster[i].set_title('%i: %s'%(i+1, self._tname[i]))
            self.im += [self.axRaster[i].imshow(self.fdata[i, tpix],
                                                extent=self.extent,
                                                origin='lower',
                                               cmap=self.cmap,
                                               interpolation=interpolation)]
            
        self.slit = self.axRaster[filterNum-1].plot([self.x1, self.x2],
                                                    [self.y1, self.y2],
                                                    color='k')[0]
        self.center = self.axRaster[filterNum-1].scatter(self.xc, self.yc,
                                                         100, marker='+',
                                                         c='k')
        self.top = self.axRaster[filterNum-1].scatter(self.x2, self.y2, 100,
                                marker='+', c='b', label='%.1f'%self.R)
        self.bottom = self.axRaster[filterNum-1].scatter(self.x1, self.y1, 100,
                                   marker='+', c='r',
                                   label='-%.1f'%self.R)
        self.tslit = self.axTD.axvline(self.t, ls='dashed', c='lime')
        self.leg = self.axRaster[filterNum-1].legend()
        self.axTD.set_aspect(adjustable='box', aspect='auto')
        self.imTD.set_clim(self.fdata[filterNum-1,0].min(),
                           self.fdata[filterNum-1,0].max())
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)
        plt.show()
        
    def _onKey(self, event):
        if event.key == 'up':
            if self.angle < 360:
                self.angle += 1
            else:
                self.angle = 1
        elif event.key == 'down':
            if self.angle > 0:
                self.angle -=1
            else:
                self.angle = 359
        elif event.key == 'right':
            if self.t < self._tarr[-1]:
                self.t += self.dt
            else:
                self.t = self._tarr[0]
        elif event.key == 'left':
            if self.t > self._tarr[0]:
                self.t -= self.dt
            else:
                self.t = self._tarr[-1]
        elif event.key == 'ctrl+right':
            if self.xc < self._xarr[-1]:
                self.xc += self.dx
            else:
                self.xc = self._xarr[0]
        elif event.key == 'ctrl+left':
            if self.xc > self._xarr[0]:
                self.xc -= self.dx
            else:
                self.xc = self._xarr[-1]
        elif event.key == 'ctrl+up':
            if self.yc < self._yarr[-1]:
                self.yc += self.dy
            else:
                self.yc = self._yarr[0]
        elif event.key == 'ctrl+down':
            if self.yc > self._yarr[0]:
                self.yc -= self.dy
            else:
                self.yc = self._yarr[-1]
        elif event.key == 'ctrl++':
            self.R += self.dx
        elif event.key == 'ctrl+-':
            self.R -= self.dx
        elif event.key == ' ' and event.inaxes in self.axRaster:
            self.xc = event.xdata
            self.yc = event.ydata
        elif event.key == ' ' and event.inaxes == self.axTD:
            self.t = event.xdata
        elif event.key == 'x' and event.inaxes == self.axTD:
            self.pos += [event.ydata]
            ang = np.deg2rad(self.angle)
            xp = self.pos[-1]*np.cos(ang) + self.xc
            yp = self.pos[-1]*np.sin(ang) + self.yc
            self.mark += [self.axRaster[self.filterNum-1].scatter(xp, yp, 100,
                                                                  marker='+',
                                                                  c='lime')]
            self.hlines += [self.axTD.axhline(self.pos[-1], ls='dashed', c='lime')]
        elif event.key == 'enter':
            if self.pause == 'ini':
                self.ani = FuncAnimation(self.fig, self._chTime,
                                         frames=self._tarr,
                                         blit=False,
                                         interval=1e3/self.fps,
                                         repeat=True)
#                                         cache_frame_data=False)
                self.pause = False
            else:
                self.pause ^= True
                if self.pause:
                    self.ani.event_source.stop()
                else:
                    self.ani.event_source.start(1e3/self.fps)
        for iid in range(self.nfilter):
            if event.key == 'ctrl+%i'%(iid+1):
                self.filterNum = iid+1
                tpix = np.abs(self._tarr-self.t).argmin()
                self.changeSlit(self.R, self.xc, self.yc, self.angle)
                self.axTD.set_title('%i: %s,  '
                            'Time: %s, '
                            'tpix: %i'%(self.filterNum, self._tname[self.filterNum-1],
                                        self.Time[tpix].value,
                                        tpix))
                self._filterNum0 = self.filterNum
                self.imTD.set_clim(self.im[self.filterNum-1].get_clim())
        
        if self.xc != self._xc0 or self.yc != self._yc0 or \
            self.angle != self._angle0 or self.R != self._R0:
                self.changeSlit(self.R, self.xc, self.yc, self.angle)
                self._R0 = self.R
                self._xc0 = self.xc
                self._yc0 = self.yc
                self._angle0 = self.angle
        if self.t != self._t0:
            self._chTime(self.t)
            self._t0 = self.t
        self.fig.canvas.draw_idle()
        
    def changeSlit(self, R, xc, yc, angle):
        td = self.get_TD(R, xc, yc, angle)
        self.tdextent[2] = -R
        self.tdextent[3] = R
        self.axTD.set_ylim(-R, R)       
        ang = np.deg2rad(self.angle)
        if self.filterNum != self._filterNum0:
            self.leg.remove()
            self.slit.remove()
            self.bottom.remove()
            self.center.remove()
            self.top.remove()
            self.slit = self.axRaster[self.filterNum-1].plot([self.x1, self.x2],
                                                             [self.y1, self.y2],
                                                             color='k')[0]
            self.center = self.axRaster[self.filterNum-1].scatter(self.xc,
                                   self.yc, 100, marker='+', c='k')
            self.top = self.axRaster[self.filterNum-1].scatter(self.x2,
                                    self.y2, 100,
                                    marker='+', c='b', label='%.1f'%self.R)
            self.bottom = self.axRaster[self.filterNum-1].scatter(self.x1,
                                       self.y1, 100,
                                       marker='+', c='r',
                                       label='-%.1f'%self.R)
            for n, pos in enumerate(self.pos):
                self.mark[n].remove()
                xp = pos*np.cos(ang) + self.xc
                yp = pos*np.sin(ang) + self.yc
                self.mark[n] = self.axRaster[self.filterNum-1].scatter(xp, yp, 100,
                                                                  marker='+',
                                                                  c='lime')
        else:
            self.slit.set_xdata([self.x1, self.x2])
            self.slit.set_ydata([self.y1, self.y2])
            self.bottom.set_offsets([self.x1, self.y1])
            self.top.set_offsets([self.x2, self.y2])
            self.center.set_offsets([self.xc, self.yc])
            # change marker
            for n, pos in enumerate(self.pos):
                xp = pos*np.cos(ang) + self.xc
                yp = pos*np.sin(ang) + self.yc
                self.mark[n].set_offsets([xp, yp])
                self.hlines[n].set_ydata(pos)
        self.top.set_label('%.1f'%self.R)
        self.bottom.set_label('-%.1f'%self.R)
        self.imTD.set_data(td)
        self.leg = self.axRaster[self.filterNum-1].legend()
        
    def _chTime(self, t):
        self.t = t
        tpix = np.abs(self._tarr-t).argmin()
        self.axTD.set_title('%i: %s,  '
                            'Time: %s, '
                            'tpix: %i'%(self.filterNum, self._tname[self.filterNum-1],
                                        self.Time[tpix].value,
                                        tpix))
        self.tslit.set_xdata(self.t)
        for n, im in enumerate(self.im):
            im.set_data(self.fdata[n, tpix])
    
    def set_clim(self, cmin, cmax, frame):
        self.im[frame-1].set_clim(cmin, cmax)
        if self.filterNum == frame:
            self.imTD.set_clim(cmin, cmax)
            
    def remove_Mark(self):
        for n in range(len(self.pos)):
            self.mark[n].remove()
            self.hlines[n].remove()
        self.pos = []
        self.mark = []
        self.hlines = []

    def savefig(self, filename, **kwargs):
        self.fig.save(filename, **kwargs)
        
    def saveani(self, filename, **kwargs):
        fps = kwargs.pop('fps', self.fps)
        self.ani.save(filename, fps=fps, **kwargs)