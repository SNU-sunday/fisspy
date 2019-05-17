"""
"""

from __future__ import absolute_import, division
import numpy as np
from interpolation.splines import LinearSpline
import matplotlib.pyplot as plt
__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"


class TDmap(object):
    def __init__(self, data, R, angle=0, extent=[0, 'end', 0, 'end'], xc=0, yc=0):
        """
        Make interactive time distance map
        
        Parameters
        ----------
        data : `~numpy.ndarray`
            3-dimensional data array, (time, y, x)
        R : `float`
            The half length of the slit
        angle : `float`, optional
            The angle of the slit in degree. Default is 0 
        extent : `list`, optional
            The bounding box in data coordinates that the image will fill.
        xc : 'float', optional
            Center of the slit position in x-axis
        yc : 'float', optional
            Center of the slit position in y-axis
            
        Returns 
        -------
        td : `~fisspy.analysis.TDmap`
            A new time distance class object.
        
        Examples
        --------
        >>> from fisspy.analysis import TDmap
        >>> from import numpy as np
        >>> data = np.random.rand(200,250) * np.sin(np.arange(300)*np.pi/20)[:,None,None]
        >>> td = TDmap(data, 40, xc=125, yc= 100)
        >>> td.imshow(clim=[-3, 3], interpolation='bilinear')
        """
        self._mark_switch = False
        self.data = data
        self._R = R
        self._angle = angle
        self._xc = xc
        self._yc = yc
        self.R = R
        self._R0 = self.R
        self.angle = angle
        self._angle0 = angle
        self.xc = xc
        self._xc0 = xc
        self.yc = yc
        self._yc0 = yc
        nt, ny, nx = data.shape
        self.nt = nt
        self.ny = ny
        self.nx = nx
        ang = np.deg2rad(angle)
        self.ang = ang
        if extent[1] == 'end':
            extent[1] = nx-1
            extent[3] = ny-1
        self.extent = extent
        self.dx = (extent[1]-extent[0])/nx
        self.x1 = -R*np.cos(ang) + xc
        self.x2 = R*np.cos(ang) + xc
        self.y1 = -R*np.sin(ang) + yc
        self.y2 = R*np.sin(ang) + yc
        smin = [extent[2], extent[0]]
        smax = [extent[3], extent[1]]
        order = [ny, nx]
        nl = int(np.ceil(2*R/(extent[1]-extent[0])*(nx-1)))
        x = np.linspace(self.x1, self.x2, nl)
        y = np.linspace(self.y1, self.y2, nl)
        td = np.empty([nl, nt])
        for i, ta in enumerate(data):
            interp = LinearSpline(smin, smax, order, ta)
            iarr = np.array([y,x]).T
            td[:,i] = interp(iarr).T
        
        self.td = td
    
    def imshow(self, rframe=False, ts=0, te=False, **kwargs):
        """
        Display interactive image and time-distance map.
        
        Parameters
        ----------
        rframe : `int`, optional
            The reference frame. Default is self.nt//2
        ts : `int`, optional
            Start time. Default is 0
        te : `int`, optional
            End time. Default is self.nt
        kwargs : 
            
        Interactive Button
        ------------------
        'left' :
            Show previous time image.
        'right' :
            Show next time image.
        'up' :
            Increase the angle of the slit by 1.
        'down' :
            Increase the angle of the slit by 1.
        'ctrl++' :
            Change the central slit position to the right by pixel size.
        'ctrl+-' :
            Change the central slit position to the left by pixel size.
        'ctrl+h' :
            Change to the orignal setting.
        """
        self.cmap = kwargs.pop('cmap', plt.cm.RdBu_r)
        self._cmap = self.cmap
        self.clim = kwargs.get('clim', [self.data.min(), self.data.max()])
        self._clim = self.clim
        if not rframe:
            rframe = self.nt//2
        if not te:
            te = self.nt-0.5
            ts = ts-0.5
        self.tLength = te-ts
        self.dt = self.tLength/self.nt
        self.frame = rframe
        self.frame0 = rframe
        self._frame = rframe
        tdextent = [ts, te, -self.R, self.R]
        self.tdextent = tdextent
        figsize = kwargs.pop('figsize', [10,7])
        self.fig, self.ax = plt.subplots(2, 1, figsize=figsize)
        self.im = self.ax[0].imshow(self.data[rframe], self.cmap,
                         origin='lower',
                         extent=self.extent, **kwargs)
        self.slit = self.ax[0].plot([self.x1, self.x2],
                        [self.y1, self.y2], color='k')
        self.center = self.ax[0].scatter(self.xc, self.yc, 100,
                             marker='+', c='k')
        self.top = self.ax[0].scatter(self.x2, self.y2, 100,
                             marker='+', c='b', label='%.1f'%self.R)
        self.bottom = self.ax[0].scatter(self.x1, self.y1, 100,
                             marker='+', c='r', label='-%.1f'%self.R)
        self.tdMap = self.ax[1].imshow(self.td, self.cmap,
                            origin='lower',
                            extent=tdextent, **kwargs)
        self.tSlit = self.ax[1].vlines(self.frame*self.dt,-self.R,self.R,
                            linestyles='dashed')
        self.ax[0].set_xlim(self.xc-self.R-1, self.xc+self.R+1)
        self.ax[0].set_ylim(self.yc-self.R-1, self.yc+self.R+1)
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')
        self.ax[0].set_title('Image')
        self.ax[1].set_xlabel('Time')
        self.ax[1].set_ylabel('Distance')
        self.ax[1].set_title('Time-Distance Map')
        self.ax[1].set_xlim(tdextent[0], tdextent[1])
        #set.ax[1].set_position([lbrt])
        self.ax[1].set_aspect(adjustable='box', aspect='auto')
        self.ax[1].set_ylim(-self.R, self.R)
        self.ax[0].legend()
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def changeSlit(self, R, angle, xc=0, yc=0):
        """
        """
        self.R = R
        self.angle = angle
        self.ang = np.deg2rad(angle)
        self.xc = xc
        self.yc = yc
        self.x1 = -R*np.cos(self.ang) + xc
        self.x2 = R*np.cos(self.ang) + xc
        self.y1 = -R*np.sin(self.ang) + yc
        self.y2 = R*np.sin(self.ang) + yc
        
        smin = [self.extent[2], self.extent[0]]
        smax = [self.extent[3], self.extent[1]]
        order = [self.ny, self.nx]
        nl = int(np.ceil(2*R/(self.extent[1]-self.extent[0])*(self.nx-1)))
        x = np.linspace(self.x1, self.x2, nl)
        y = np.linspace(self.y1, self.y2, nl)
        td = np.empty([nl, self.nt])
        for i, ta in enumerate(self.data):
            interp = LinearSpline(smin, smax, order, ta)
            iarr = np.array([y,x]).T
            td[:,i] = interp(iarr).T
        
        self.tdextent[2] = -R
        self.tdextent[3] = R
        self.td = td
        
        self.slit.pop(0).remove()
        self.bottom.remove()
        self.top.remove()
        self.slit = self.ax[0].plot([self.x1, self.x2],
                             [self.y1, self.y2], color='k')
        self.bottom = self.ax[0].scatter(self.x1, self.y1, 100,
                             marker='+', c='r', label='-%.1f'%self.R)
        self.top = self.ax[0].scatter(self.x2, self.y2, 100,
                             marker='+', c='b', label='%.1f'%self.R)
        self.tdMap.set_data(self.td)
        if self._mark_switch:
            self._rmMark()
            self.regionMark(self.pos)
        if self.R != self._R0:
            self.tdMap.set_extent(self.tdextent)
            self.ax[1].set_ylim(-self.R, self.R)
            self.ax[0].set_xlim(self.xc-self.R-1, self.xc+self.R+1)
            self.ax[0].set_ylim(self.yc-self.R-1, self.yc+self.R+1)
            self.ax[0].legend()
#            self.ax[1].set_aspect(adjustable='box', aspect=0.8*self.R/80)
            self.fig.tight_layout()
            self.tSlit.remove()
            self.tSlit = self.ax[1].vlines(self.frame*self.dt,-self.R,self.R,
                            linestyles='dashed')
        if self.xc != self._xc0 or self.yc != self._yc0:
            self.center.remove()
            self.center = self.ax[0].scatter(self.xc, self.yc, 100,
                                 marker='+', c='k')
            self.ax[0].set_xlim(self.xc-self.R-1, self.xc+self.R+1)
            self.ax[0].set_ylim(self.yc-self.R-1, self.yc+self.R+1)
            
            
    def regionMark(self, position):
        """
        siwtch: True = on
                False = off
        position: float or list
        """
        
        self._mark_switch = True
        self.pos = np.array(position).flatten()
        self.np = len(self.pos)
        
        self.xp = self.pos*np.cos(self.ang) + self.xc
        self.yp = self.pos*np.sin(self.ang) + self.yc
        self.mark = self.ax[0].scatter(self.xp, self.yp, 100,
                           marker='x', c='k')
        self.hlines = self.ax[1].hlines(self.pos,
                             self.tdextent[0], self.tdextent[1],
                             linestyles='dashed')
    
    def chRegion(self, position):
        """
        """
        self.mark.remove()
        self.hlines.remove()
        self.regionMark(position)
        
    def _rmMark(self):
        self.mark.remove()
        self.hlines.remove()
            
    def _on_key(self, event):
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
            if self.frame < self.nt-1:
                self.frame += 1
            else:
                self.frame = 0

        elif event.key == 'left':
            if self.frame > 0:
                self.frame -=1
            else:
                self.frame = self.nt-1
        
        elif event.key == 'ctrl+right':
            if self.xc <= self.extent[1]:
                self.xc += self.dx
            else:
                self.xc = self.extent[0]
        elif event.key == 'ctrl+left':
            if self.xc >= self.extent[0]:
                self.xc -= self.dx
            else:
                self.xc = self.extent[1]
        elif event.key == 'ctrl+up':
            if self.yc <= self.extent[3]:
                self.yc += self.dx
            else:
                self.yc = self.extent[2]
        elif event.key == 'ctrl+down':
            if self.yc >= self.extent[2]:
                self.yc -= self.dx
            else:
                self.yc = self.extent[3]
        elif event.key == 'ctrl++':
            self.R += self.dx
        elif event.key == 'ctrl+-':
            self.R -= self.dx
        elif event.key == 'ctrl+h':
            self.R = self._R
            self.xc = self._xc
            self.yc = self._yc
            self.angle = self._angle
            self.frame = self._frame
            self.chclim(self._clim[0], self._clim[1])
            self.chcmap(self._cmap)
        
        if self.angle != self._angle0 or self.xc != self._xc0 or self.yc != self._yc0 or self.R != self._R0:
            self.changeSlit(self.R, self.angle, xc=self.xc, yc=self.yc)
            self._angle0 = self.angle
            self._xc0 = self.xc
            self._yc0 = self.yc
            self._R0 = self.R
        if self.frame != self.frame0:
            self.im.set_data(self.data[self.frame])
            self.tSlit.remove()
            self.tSlit = self.ax[1].vlines(self.frame*self.dt,-self.R,self.R,
                            linestyles='dashed')
            self.frame0 = self.frame
        self.fig.canvas.draw_idle()
    
    def chclim(self, cmin, cmax):
        """
        """
        self.im.set_clim(cmin, cmax)
        self.tdMap.set_clim(cmin, cmax)
        self.clim = [cmin, cmax]
        
    def chcmap(self, cmap):
        """
        """
        self.cmap=cmap
        self.im.set_cmap(cmap)
        self.tdMap.set_cmap(cmap)