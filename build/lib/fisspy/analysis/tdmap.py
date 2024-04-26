from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from interpolation.splines import LinearSpline
from scipy.interpolate import CubicSpline as CS
from os.path import join
from os import getcwd
import matplotlib.patheffects as pe

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

__all__ = ["makeTDmap", "analysisTDmap"]

class makeTDmap:
    """
    Make Time-Distance map for given slit position interactively

    Parameters
    ----------
    data : `~numpy.ndarray`
        3-dimensional data array with the shape of (nt, ny, nx).
    dx : `float` (optional)
        Pixel scale along x-axis in the unit of km.
    dy : `float` (optional)
        Pixel scale along y-axis in the unit of km.
    dt : `float` (optional)
        Pixel scale along t-axis in the unit of sec.
    cmap : matplotlib color map (optional)
        Colormap of the image.
    figsize : `list` (optional)
        Figure size.
    dpi : `int` (optional)
        Depth per inch
    clim : `list` (optional)
        Color limit of the image.

    Interactive Keys
    ----------------
    right : next frame
    left : previous frame
    spacebar : select TD slit position (npoint >= 2)
        if npoint=2, linear slit (1st point-start point, 2nd point-end point)
        if npoint=3, arc slit (1st point-center of the arc, 2nd point-start point, 3rd point-angle of the arc)
        if npoint>=4, arbtrary curve (1st point-start point, nth point-end point)
    c : calculate and show TD map on the new figure.
    ctrl+h, cmd+h: open help box
    """
    def __init__(self, data, dx=1, dy=1, dt=1, cmap=None, figsize=None, dpi=100, clim=None, label=None, aspect=1/10):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        try:
            plt.rcParams['keymap.back'].remove('c')
            plt.rcParams['keymap.forward'].remove('v')
        except:
            pass

        # set initial parameter
        self.data = data
        self.aspect = aspect
        self.nt, self.ny, self.nx = data.shape
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.clim = clim
        if cmap is None:
            Cmap = plt.cm.gray
        self.cmap = Cmap
        self.analysis = None
        self.a_fig = None
        self.a_ax = None
        self.h_fig = None
        self.t = 0
        self.time = np.arange(0, self.nt)*self.dt
        self.slitPoint = []
        self.pslitPoint = []
        self.cplot = None
        self.lplot = None
        self.onSlit = False
        # save parameter
        self.spGrad = []
        self.sv = []
        self.sfit = []
        self.Tperiod = []
        self.Tline_pos = []
        self.Ddistance = []
        self.Dline_pos = []

        # figure window label
        afl = plt.get_figlabels()
        if label is None:
            Label = 'Unknown'
        if Label in afl:
            ii = 0
            tmp2 = []
            for aa in afl:
                if aa.find('-TD') == -1:
                    continue
                tmp = aa.split(f'{Label}_case')
                if len(tmp) > 1:
                    ii += 1
                    tmp2 += [int(tmp[-1])]
            if ii == 0:
                Label = Label+'_case2'
            else:
                Label = Label+f'_caseq{max(tmp2)+1}'
        self.label = Label

        # make interpolation function
        smin = [0, 0, 0]
        smax = [(self.nt-1)*dt, (self.ny-1)*dy, (self.nx-1)*dx]
        order = data.shape
        self.interp = LinearSpline(smin, smax, order, data)

        # figure setting
        self.outline = [pe.Stroke(linewidth=plt.rcParams['lines.linewidth']*2, foreground='k', alpha=0.3), pe.Normal()]
        wratio = self.ny/self.nx
        l = -dx*0.5
        r = (self.nx - 0.5)*dx
        b = -dy*0.5
        t = (self.ny - 0.5)*dy
        self.extent = [l, r, b, t]
        
        if figsize is None:
            fs = [8,8*wratio+0.2]

        self.fig, self.ax = plt.subplots(figsize=fs, dpi=dpi, num=self.label)
        self.im = self.ax.imshow(self.data[self.t], self.cmap, extent=self.extent, origin='lower')
        if self.clim is not None:
            self.im.set_clim(self.clim)
        self.ax.set_title('t = 0 (sec)')
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)
        self.fig.canvas.mpl_connect('motion_notify_event', self._circleHelp)
        self.fig.tight_layout()
        self.fig.show()

    def _circleHelp(self, event):
        if event.inaxes == self.ax and len(self.slitPoint) == 1:
            if self.lplot is not None:
                for ii in self.lplot:
                    ii.remove()
            self.lplot = []
            x1, y1 = self.slitPoint[0]
            x2 = event.xdata
            y2 = event.ydata
            self.lplot += [self.ax.plot([x1,x2], [y1,y2], color='r', ls='dashed', alpha=0.3, path_effects=self.outline)[0]]
            self.fig.canvas.draw_idle()
        elif event.inaxes == self.ax and len(self.slitPoint) == 2:
            if self.cplot is not None:
                for ii in self.cplot:
                    ii.remove()
            self.cplot = []
            xc, yc = self.slitPoint[0]
            x1, y1 = self.slitPoint[1]
            x2 = event.xdata
            y2 = event.ydata
            r = np.sqrt((x1-xc)**2 + (y1-yc)**2)
            r2 = np.sqrt((x2-xc)**2 + (y2-yc)**2)
            theta = np.arccos(((x2-xc)*(x1-xc)+(y2-yc)*(y1-yc))/(r*r2))
            a1 = (y1-yc)/(x1-xc)
            yint2 = y2-a1*x2
            yint1 = y1-a1*x1
            if (yint2 < yint1 and x1>=xc) or (yint2 > yint1 and x1<xc):
                theta *= -1
            # sign = np.arctan2(y2-y1,x2-x1)
            # if sign < 0:
            #     theta *=-1
            tt = np.arctan2(y2-yc,x2-xc)
            t1 = np.arctan2(y1-yc,x1-xc)
            sign = -1 if (tt >= np.pi/2 and tt <= np.pi) or (tt < -np.pi/2 and tt > -np.pi) else 1
            a = (y2-yc)/(x2-xc)
            xe = sign*np.sqrt(r**2/(1+a**2))+xc
            ye = a*(xe-xc)+yc

            atheta = np.linspace(0, theta)
            x = xc + r*np.cos(atheta+t1)
            y = yc + r*np.sin(atheta+t1)
            self.cplot += [self.ax.plot([xc,x1], [yc,y1], color='k', ls='dashed', alpha=0.3)[0]]
            self.cplot += [self.ax.plot([xc,xe], [yc,ye], color='k', ls='dashed', alpha=0.3)[0]]
            self.cplot += [self.ax.plot(x, y, color='r', ls='dashed', alpha=0.3, path_effects=self.outline)[0]]
            self.fig.canvas.draw_idle()

    def makeTD(self, sp):
        """
        Make TD map for a given slit position.
        
        sp: `list`
            slit position.
        """
        nsp = len(sp)
        if nsp == 1:
            raise ValueError("The number of slit point should be larger than two.")
        elif nsp == 2:
            x1, y1 = sp[0]
            x2, y2 = sp[1]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            nl = int(length/self.dx)*2
            
            x = np.linspace(x1, x2, nl)[None,:]*np.ones((self.nt,nl))
            y = np.linspace(y1, y2, nl)[None,:]*np.ones((self.nt,nl))
            self.xslit = np.array([x1, x2])
            self.yslit = np.array([y1, y2])
        elif nsp == 3:
            xc, yc = sp[0]
            x1, y1 = sp[1]
            x2, y2 = sp[2]
            r = np.sqrt((x1-xc)**2 + (y1-yc)**2)
            r2 = np.sqrt((x2-xc)**2 + (y2-yc)**2)
            t1 = np.arctan2(y1-yc,x1-xc)
            theta = np.arccos(((x2-xc)*(x1-xc)+(y2-yc)*(y1-yc))/(r*r2))
            a1 = (y1-yc)/(x1-xc)
            yint2 = y2-a1*x2
            yint1 = y1-a1*x1
            if (yint2 < yint1 and x1>=xc) or (yint2 > yint1 and x1<xc):
                theta *= -1
          
            length = abs(theta*r)
            nl = int(length/self.dx)*2
            atheta = np.linspace(0, theta, nl)
            x = xc + r*np.cos(atheta+t1)[None,:]*np.ones((self.nt,nl))
            y = yc + r*np.sin(atheta+t1)[None,:]*np.ones((self.nt,nl))
            self.xslit = x[0]
            self.yslit = y[0]
        else:
            x = np.zeros(nsp)
            y = np.zeros(nsp)
            for i,ss in enumerate(sp):
                x[i] = ss[0]
                y[i] = ss[1]
            theta = np.arctan2(y[-1]-y[0],x[-1]-x[0])
            xt = x*np.cos(theta) + y*np.sin(theta)
            yt = -x*np.sin(theta) + y*np.cos(theta)
            cs = CS(xt, yt)
            seg = np.linspace(xt[0], xt[-1], 2000)
            yseg = cs(seg)
            ds = np.sqrt((np.roll(seg,-1) - seg)**2 + (np.roll(yseg,-1)-yseg)**2)
            length = ds[:-1].sum()
            nl = int(length/self.dx)*2
            dl = length/nl
            cl = ds[:-1].cumsum()
            self.cl = cl
            lcs = CS(cl, seg[1:])
            xxt = np.zeros(nl)
            il = np.linspace(0, length, nl)
            xxt[0] = xt[0]
            xxt[1:] = lcs(il[1:])
            yyt = cs(xxt)
            x = xxt*np.cos(theta) - yyt*np.sin(theta)
            y = xxt*np.sin(theta) + yyt*np.cos(theta)
            x = x[None,:]*np.ones((self.nt, nl))
            y = y[None,:]*np.ones((self.nt, nl))
            self.xslit = seg*np.cos(theta) - yseg*np.sin(theta)
            self.yslit = seg*np.sin(theta) + yseg*np.cos(theta)

        self.dl = length/nl
        inp = np.array([(self.time[:,None] * np.ones((self.nt,nl))).flatten(), y.flatten(), x.flatten()])
        return self.interp(inp.T).reshape((self.nt,nl)).T

    def clear_marker(self):
        if len(self.slitPoint) == 3:
            self.pslitPoint[2].remove()
            self.pslitPoint[0].set_color('b')
            self.pslitPoint[0].set_marker('x')
            self.pslitPoint[1].set_color('r')
            self.pslitPoint[1].set_marker('+')
        else:
            for p in self.pslitPoint[1:]:
                p.remove()

    def clear_slit(self):
        lines = self.ax.get_lines()
        for l in lines:
            l.remove()
        self.fig.canvas.draw_idle()

    def _onKey(self, event):
        if event.key == 'left':
            self._prev()
        elif event.key == 'right':
            self._next()
        elif event.key == ' ' and event.inaxes == self.ax:
            if self.onSlit:
                self.clear_slit()
                self.onSlit = False
            # point to make tdmap slit
            self.slitPoint += [[event.xdata, event.ydata]]
            if len(self.slitPoint) == 1:
                c = 'r'
            else:
                c = 'lime'
            if len(self.slitPoint) == 2:
                if self.lplot is not None:
                    for ii in self.lplot:
                        ii.remove()
                self.lplot = None
            if len(self.slitPoint) == 3:
                if self.cplot is not None:
                    for ii in self.cplot:
                        ii.remove()
                self.cplot = None
            self.pslitPoint += [self.ax.plot(self.slitPoint[-1][0], self.slitPoint[-1][1], '+', color=c)[0]]
            self.fig.canvas.draw_idle()
        elif event.key == 'c' and event.inaxes == self.ax:
            if self.lplot is not None:
                    for ii in self.lplot:
                        ii.remove()
            if self.cplot is not None:
                    for ii in self.cplot:
                        ii.remove()
            self.lplot = None
            self.cplot = None
            # make td map
            self.TD = self.makeTD(self.slitPoint)
            if self.a_fig is not None:
                plt.close(self.a_fig)
            self.analysis = analysisTDmap(self.TD, self.dl, self.dt, cmap=self.cmap, parent=self, clim=self.clim, t=self.t, label=self.label, aspect=self.aspect)
            self.a_fig = self.analysis.fig
            self.a_ax = self.analysis.ax
            self.slit = self.ax.plot(self.xslit, self.yslit, color='lime')[0]
            self.clear_marker()
            self.pslitPoint = []
            self.slitPoint = []
            self.onSlit = True
            self.fig.canvas.draw_idle()
        elif event.key == 'escape':
            if len(self.slitPoint) != 0:
                if self.lplot is not None:
                    if len(self.lplot) >= 1:
                        self.lplot.pop(-1).remove()
                self.lplot = []
                if self.cplot is not None:
                    for ii in self.cplot:
                        ii.remove()
                self.cplot = None
                self.pslitPoint.pop(-1).remove()
                self.slitPoint.pop(-1)
                self.fig.canvas.draw_idle()
            if self.onSlit:
                self.clear_slit()
                self.onSlit = False
        elif (event.key == 'cmd+h' or event.key == 'ctrl+h'):
            if self.h_fig is not None:
                plt.close(self.h_fig)
                self.h_fig = None
            else:
                tm = 0.155
                lm = 0.03
                dh = 0.06
                self.h_fig = plt.figure(num='Help box for MTD', figsize=[5,4], facecolor='linen')
                self.h_fig.text(0.5, 1-tm+2*dh, '<Interactive keys>', ha='center', va='top', size=15, weight='bold')
                self.h_fig.text(lm, 1-tm, 'left: Previous frame', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-dh, 'right: Next frame', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-2*dh, 'spacebar: Select TD slit position (npoints should be >=2)\n    * if npoints=2, make linear slit\n        (1st: start point, 2nd: end point)\n    * if npoints=3, make arc-shaped slit\n        (1st=center, 2nd: start point, 3rd: endpoint)\n    * if npoints>3, make arbitrary curve by interpolation\n        (1st: start point, n-th: end point)', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-8*dh, 'c: Create TD map', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-9*dh, 'esc: erase the last slit position\n  or if you draw the slit already erase the slit', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-10.8*dh, 'cmd+h or ctrl+h: open the help box figure', ha='left', va='top', size=12)
                self.h_fig.show()

    def _next(self):
        if self.t < self.nt-1:
            self.t += 1
        elif self.t >= self.nt-1:
            self.t = 0
        self.chTime(self.t)

    def _prev(self):
        if self.t > 0:
            self.t -= 1
        elif self.t <= 0:
            self.t = self.nt-1
        self.chTime(self.t)
    
    def chTime(self, t):
        self.t = t
        self.im.set_data(self.data[self.t])
        self.ax.set_title(f't = {self.t*self.dt:.2f} (sec)')
        if self.analysis is not None:
            self.analysis.t = self.t
            self.analysis.chTime()
        self.fig.canvas.draw_idle()

    def save(self, fname=None):
        """
        extension should be npz
        """
        if fname is None:
            fname = join(getcwd(), f"{self.label}.npz")
        if fname.split('.')[-1] != 'npz':
            raise ValueError("File extension should be npz.")
        if self.analysis is None:
            np.savez(fname, TD=self.TD, Slit=[self.xslit, self.yslit], dl=self.dl, dx=self.dx, dy=self.dy, dt=self.dt)
        else:
            np.savez(fname, TD=self.TD, Slit=[self.xslit, self.yslit], dl=self.dl, dx=self.dx, dy=self.dy, dt=self.dt, vposition=self.spGrad, velocity=self.sv, boolFit=self.sfit, period=self.Tperiod, pposition=self.Tline_pos, wavelength=self.Ddistance, wposition=self.Dline_pos)

class analysisTDmap:
    def __init__(self, TD, dl, dt, cmap=None, figsize=[19, 8], dpi=100, parent=None, clim=None, t=0, label=None, aspect=1/10):
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass
        try:
            plt.rcParams['keymap.back'].remove('c')
            plt.rcParams['keymap.forward'].remove('v')
        except:
            pass
        self.TD = TD
        self.dl = dl
        self.dt = dt
        self.nl, self.nt = TD.shape
        self.clim = clim
        self.parent = parent

        self.time = np.arange(0, self.nt)*self.dt
        self.distance = np.arange(0, self.nl)*self.dl
        self.t = t
        self.r = 0
        self.idx = False
        self.idxT = False
        self.idxD = False
        self.marker = [None]*2
        self.point = [None]*2
        self.markerT = [None]*2
        self.pointT = [None]*2
        self.markerD = [None]*2
        self.pointD = [None]*2
        self.slits = []
        self.vtexts = []
        self.tpp = []
        self.spGrad = []
        self.sv = []
        self.sfit = []
        self.Tperiod = []
        self.Tlines = []
        self.Tline_pos = []
        self.pTexts = []
        self.Ddistance = []
        self.Dlines = []
        self.Dline_pos = []
        self.dTexts = []
        self.h_fig = None

        # figure window label
        afl = plt.get_figlabels()
        if label is None:
            Label = 'Unknown-TD'
        else:
            Label = Label+'-TD'
        if Label in afl:
            l0 = Label.split('-TD')[0]
            ii = 0
            tmp2 = []
            for aa in afl:
                tmp = aa.split(f'{l0}_case')
                if len(tmp) > 1:
                    ii += 1
                    tmp2 += [int(tmp[-1])]
            if ii == 0:
                Label = l0+'_case2-TD'
            else:
                print(1)
                Label = l0+f'_case{max(tmp2)+1}-TD'
        self.label = Label
        l = -0.5*dt
        r = (self.nt - 0.5)*dt
        b = -dl*0.5
        t = (self.nl - 0.5)*dl
        extent = [l, r, b, t]

        # figsize = [19,8]
        fx, fy = figsize
        fratio = figsize[1]/figsize[0]
        # xm = ymargin = 0.07
        # ym = xmargin = (ymargin+0.02) * fratio

        xM = 0.8
        yM = xM*4/5
        xm = xmargin = xM/fx
        
        kx = 8/9
        ky = 2/3
        yl = t-b
        xl = r-l
        xs = kx*fx*(1-2.5*xm)
        ys = xs*yl*aspect/xl
        fy = ys/ky+ 2.5*yM

        if fy > 10:
            fy = 10
            ym = ymargin = yM/fy

            ys2 = ky*fy*(1-2.5*ym)
            xs2 = ys2*xl/(aspect*yl)
            fx = xs2/kx+ 2.5*xM
            xm = xmargin = xM/fx
        ym = ymargin = yM/fy
        self.fig, self.ax = plt.subplots(figsize=[fx,fy], dpi=dpi, label=self.label)

        self.ax.set_position([xmargin, ymargin, (1-2.5*xmargin)/9*8, (1-2.5*ymargin)/3*2])
        self.axT = self.fig.add_subplot(111, sharex=self.ax)
        self.axT.set_position([xmargin, ymargin*2+(1-2.5*ymargin)/3*2, (1-2.5*xmargin)/9*8, (1-2.5*ymargin)/3])
        self.axD = self.fig.add_subplot(111, sharey=self.ax)
        self.axD.set_position([xmargin*2+(1-2.5*xmargin)/9*8, ymargin, (1-2.5*xmargin)/9, (1-2.5*ymargin)/3*2])
        self.imTD = self.ax.imshow(TD, cmap, origin='lower', extent=extent, aspect=aspect)
        if self.clim is not None:
            self.imTD.set_clim(self.clim)
        self.clim = self.imTD.get_clim()
        # self.ax.set_aspect(aspect='auto', adjustable='box')
        self.ax.set_xlabel('Time (sec)')
        self.ax.set_ylabel('Distance (km)')
        self.TS = self.axT.plot(self.time, TD[self.r], color='k')[0]
        self.DS = self.axD.plot(TD[:,self.t], self.distance, color='k')[0]
        self.axD.set_xlim(self.TD.min(), self.TD.max())
        self.axT.set_ylim(self.TD.min(), self.TD.max())
        self.hline = self.ax.plot([l, r], [self.r*self.dl, self.r*self.dl], color='darkcyan', ls='dashed', dashes=(5, 10))[0]
        self.hline_sub = self.axD.plot(self.axD.get_xlim(), [self.r*self.dl, self.r*self.dl], color='darkcyan', ls='dashed', dashes=(5, 10))[0]
        self.vline = self.ax.plot([self.t*self.dt, self.t*self.dt], [b, t], color='darkcyan', ls='dashed', dashes=(5, 10))[0]
        self.vline_sub = self.axT.plot([self.t*self.dt, self.t*self.dt], self.axT.get_ylim(), color='darkcyan', ls='dashed', dashes=(5, 10))[0]
        self.ax.set_xlim(l, r)
        self.ax.set_ylim(b, t)

        # x and y axis lines
        m = (self.clim[0]+self.clim[1])/2
        self.axT.plot([l,r], [m,m], color='gray', ls='dashed')
        self.axD.plot([m,m], [b,t], color='gray', ls='dashed')

        # velocity slope
        xticks = self.ax.get_xticks()
        dx = xticks[1] - xticks[0]
        xm = np.abs([xticks[0], xticks[-1]]).max()
        xticks = np.arange(-xm,xm,dx)
        nxt = len(xticks)
        self.glines = [None]*nxt
        tt = np.array([0, self.time[-1]])
        for i, xt in enumerate(xticks):
            tt[0] = xt
            dd = (tt-xt)/aspect
            self.glines[i] = self.ax.plot(tt, dd, ls='dashed', color='silver')[0]
            self.glines[i].set_visible(False)

        self.fig.canvas.mpl_connect('key_press_event', self._onKey)
        self.fig.show()
        
    def clear_marker(self):
        if self.marker[1] is not None:
            for mm in self.marker:
                mm.remove()
            self.marker = [None]*2
            self.point = [None]*2
            self.fig.canvas.draw_idle()

    def clear_markerT(self):
        if self.markerT[1] is not None:
            for mm in self.markerT:
                mm.remove()
            self.markerT = [None]*2
            self.pointT = [None]*2
            self.fig.canvas.draw_idle()

    def clear_markerD(self):
        if self.markerD[1] is not None:
            for mm in self.markerD:
                mm.remove()
            self.markerD = [None]*2
            self.pointD = [None]*2
            self.fig.canvas.draw_idle()

    def clear_slit_text(self):
        for ss in self.slits:
            ss.remove()
        for tt in self.vtexts:
            tt.remove()
        for tmp in self.tpp:
            tmp.remove()
        self.slits = []
        self.vtexts = []
        self.tpp = []
        self.fig.canvas.draw_idle()

    def calTDvel(self):
        p1, p2 = self.point

        x1, y1 = p1
        x2, y2 = p2

        yp1 = int(y1/self.dl)
        xp1 = int(x1/self.dt)
        yp2 = int(y2/self.dl)
        xp2 = int(x2/self.dt)

        a = (yp2-yp1)/(xp2-xp1)
        sign = np.sign(yp2-yp1)
        
        yy = np.arange(yp1, yp2+sign, sign)
        xx = np.zeros(len(yy))
        for ii, yi in enumerate(yy):
            xi = int(round((yi-yp1)/a+xp1))
            dd = self.TD[yi,xi-2:xi+3]
            p = np.polyfit(np.arange(xi-2,xi+3), dd, 2)
            xi2 = -p[1]/2/p[0]
            if abs(xi2-xi) > 3.5:
                xi2 = xi
            xx[ii] = xi2

        ps = np.polyfit(xx, yy, 1)
        self.v = ps[0]*self.dl/self.dt
        xf = np.array([xp1, xp2])
        yf = np.polyval(ps, xf)*self.dl
        xf *= self.dt

        self.clear_marker()
        self.tpp += [self.ax.plot(xx*self.dt, yy*self.dl, 'bx')[0]] # test
        self.slits += [self.ax.plot(xf, yf, color='lime')[0]]
        self.vtexts += [self.ax.text(xf.max()+self.dt*2, yf[xf.argmax()]+sign*self.dl*2, f'{self.v:.2f} km/s', ha='left', color='k', bbox=dict(boxstyle="round",ec='none',fc='w', alpha=0.3))]
        self.spGrad += [[xf, yf]]
        self.sv += [self.v]
        self.sfit += [True]
        if self.parent is not None:
            self.parent.spGrad = self.spGrad
            self.parent.sv = self.sv
            self.parent.sfit = self.sfit
        self.fig.canvas.draw_idle()

    def calTDvel_simple(self):
        p1, p2 = self.point
        x1, y1 = p1
        x2, y2 = p2
        xf = np.array([x1,x2])
        yf = np.array([y1,y2])
        a = (y2-y1)/(x2-x1)
        sign = np.sign(a)
                
        self.clear_marker()
        self.slits += [self.ax.plot(xf, yf, color='cyan')[0]]
        self.vtexts += [self.ax.text(xf.max()+self.dt*2, yf[xf.argmax()]+sign*self.dl*2, f'{a:.2f} km/s', ha='left', color='k', bbox=dict(boxstyle="round",ec='none',fc='w', alpha=0.3))]
        self.tpp += [None]
        self.spGrad += [[xf, yf]]
        self.sv += [a]
        self.sfit += [False]
        if self.parent is not None:
            self.parent.spGrad = self.spGrad
            self.parent.sv = self.sv
            self.parent.sfit = self.sfit
        self.fig.canvas.draw_idle()

    def calPeriod(self):
        t1, t2 = self.pointT
        tf = np.array([t1,t2])
        self.clear_markerT()

        period = abs(t2 - t1)
        yy = (self.clim[0] + self.clim[1])/2
        amp = abs(self.clim[0] - self.clim[1])
        yerr = amp*0.05
        self.Tperiod += [period]
        self.Tlines += [self.axT.errorbar(tf, [yy, yy], yerr=[yerr,yerr], color='b')]
        self.pTexts += [self.axT.text(tf.mean(), yy-amp*0.2, f'{period:.1f} sec', ha='center', color='k', bbox=dict(boxstyle="round",ec='none',fc='w', alpha=0.8))]
        self.Tline_pos += [tf]
        if self.parent is not None:
                self.parent.Tperiod = self.Tperiod
                self.parent.Tline_pos = self.Tline_pos
        self.fig.canvas.draw_idle()

    def calDistance(self):
        d1, d2 = self.pointD
        df = np.array([d1,d2])
        self.clear_markerD()

        distance = abs(d2 - d1)
        xx = (self.clim[0] + self.clim[1])/2
        amp = abs(self.clim[0] - self.clim[1])
        xerr = amp*0.05
        self.Ddistance += [distance]
        self.Dlines += [self.axD.errorbar([xx, xx], df, xerr=[xerr,xerr], color='b')]
        self.dTexts += [self.axD.text(xx+amp*0.1, df.mean(), f'{distance:.1f} km', ha='left', color='k', bbox=dict(boxstyle="round",ec='none',fc='w', alpha=0.8))]
        self.Dline_pos += [df]
        if self.parent is not None:
                self.parent.Ddistance = self.Ddistance
                self.parent.Dline_pos = self.Dline_pos
        self.fig.canvas.draw_idle()

    def _onKey(self, event):
        if event.key == 'left':
            self._prev()
        elif event.key == 'right':
            self._next()
        elif event.key == 'down':
            self._down()
        elif event.key == 'up':
            self._up()
        elif event.key == ' ' and event.inaxes == self.ax:
            self.point[self.idx] = [event.xdata, event.ydata]
            if self.marker[self.idx] is None:
                self.marker[self.idx] = self.ax.plot(event.xdata, event.ydata, 'x', color='lime')[0]
            else:
                self.marker[self.idx].set_xdata(event.xdata)
                self.marker[self.idx].set_ydata(event.ydata)
            self.idx ^= True
            self.fig.canvas.draw_idle()
        elif event.key == ' ' and event.inaxes == self.axT:
            self.pointT[self.idxT] = event.xdata
            if self.markerT[self.idxT] is None:
                self.markerT[self.idxT] = self.axT.plot(event.xdata, event.ydata, 'x', color='lime')[0]
            else:
                self.markerT[self.idxT].set_xdata(event.xdata)
                self.markerT[self.idxT].set_ydata(event.ydata)
            self.idxT ^= True
            self.fig.canvas.draw_idle()
        elif event.key == ' ' and event.inaxes == self.axD:
            self.pointD[self.idxD] = event.ydata
            if self.markerD[self.idxD] is None:
                self.markerD[self.idxD] = self.axD.plot(event.xdata, event.ydata, 'x', color='lime')[0]
            else:
                self.markerD[self.idxD].set_xdata(event.xdata)
                self.markerD[self.idxD].set_ydata(event.ydata)
            self.idxD ^= True
            self.fig.canvas.draw_idle()
        elif event.key == 'c' and self.point[1] is not None:
            self.calTDvel()
        elif event.key == 'v' and self.point[1] is not None:
            self.calTDvel_simple()
        elif (event.key == 'c' or event.key == 'v') and self.pointT[1] is not None:
            self.calPeriod()
        elif (event.key == 'c' or event.key == 'v') and self.pointD[1] is not None:
            self.calDistance()
        elif event.key == 'escape' and event.inaxes == self.ax:
            if len(self.slits) >= 1:
                self.slits.pop(-1).remove()
                self.vtexts.pop(-1).remove()
                self.spGrad.pop(-1)
                self.sfit.pop(-1)
                self.sv.pop(-1)
                tmp = self.tpp.pop(-1)
                if tmp is not None:
                    tmp.remove()
                if self.parent is not None:
                    self.parent.spGrad = self.spGrad
                    self.parent.sv = self.sv
                    self.parent.sfit = self.sfit
                self.fig.canvas.draw_idle()
        elif event.key == 'escape' and event.inaxes == self.axT:
            if len(self.Tperiod) >= 1:
                self.Tperiod.pop(-1)
                self.Tline_pos.pop(-1)
                self.Tlines.pop(-1).remove()
                self.pTexts.pop(-1).remove()
                if self.parent is not None:
                    self.parent.Tperiod = self.Tperiod
                    self.parent.Tline_pos = self.Tline_pos
                self.fig.canvas.draw_idle()
        elif event.key == 'escape' and event.inaxes == self.axD:
            if len(self.Ddistance) >= 1:
                self.Ddistance.pop(-1)
                self.Dline_pos.pop(-1)
                self.Dlines.pop(-1).remove()
                self.dTexts.pop(-1).remove()
                if self.parent is not None:
                    self.parent.Ddistance = self.Ddistance
                    self.parent.Dline_pos = self.Dline_pos
                self.fig.canvas.draw_idle()
        elif (event.key == 'cmd+r' or event.key == 'ctrl+r') and event.inaxes == self.ax:
            for i in range(len(self.slits)):
                self.slits.pop(-1).remove()
                self.vtexts.pop(-1).remove()
                self.spGrad.pop(-1)
                self.sfit.pop(-1)
                self.sv.pop(-1)
                tmp = self.tpp.pop(-1)
                if tmp is not None:
                    tmp.remove()
            if self.parent is not None:
                self.parent.spGrad = self.spGrad
                self.parent.sv = self.sv
                self.parent.sfit = self.sfit
            self.fig.canvas.draw_idle()
        elif (event.key == 'cmd+r' or event.key == 'ctrl+r') and event.inaxes == self.axT:
            for i in  range(len(self.Tperiod)):
                self.Tperiod.pop(-1)
                self.Tline_pos.pop(-1)
                self.Tlines.pop(-1).remove()
                self.pTexts.pop(-1).remove()
            if self.parent is not None:
                self.parent.Tperiod = self.Tperiod
                self.parent.Tline_pos = self.Tline_pos
            self.fig.canvas.draw_idle()
        elif (event.key == 'cmd+r' or event.key == 'ctrl+r') and event.inaxes == self.axD:
            for i in range(len(self.Ddistance)):
                self.Ddistance.pop(-1)
                self.Dline_pos.pop(-1)
                self.Dlines.pop(-1).remove()
                self.dTexts.pop(-1).remove()
            if self.parent is not None:
                self.parent.Ddistance = self.Ddistance
                self.parent.Dline_pos = self.Dline_pos
            self.fig.canvas.draw_idle()
        elif (event.key == 'cmd+h' or event.key == 'ctrl+h'):
            if self.h_fig is not None:
                plt.close(self.h_fig)
                self.h_fig = None
            else:
                tm = 0.155
                lm = 0.03
                dh = 0.06
                self.h_fig = plt.figure(num='Help box for ATD', figsize=[5,4], facecolor='azure')
                self.h_fig.text(0.5, 1-tm+2*dh, '<Interactive keys>', ha='center', va='top', size=15, weight='bold')
                self.h_fig.text(lm, 1-tm, 'left: Move left the vertical line', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-dh, 'right: Move right the vertical line', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-2*dh, 'bottom: Move down the horizontal line', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-3*dh, 'top: Move up the horizontal line', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-4*dh, 'spacebar on each axes: Mark the position\n                                     (need 2 position)', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-5.8*dh, 'c on TDmap: Calculate the gradient fitting the peak\n                     values of the ridges between two marked\n                     positions.', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-8.4*dh, 'v on any axes: Simply calculate the measurement\n                        between two marked positions.', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-10.2*dh, 'esc on each axes: Remove the last measurement', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-11.2*dh, 'ctrl+r or cmd+r: Remove the all measurement', ha='left', va='top', size=12)
                self.h_fig.text(lm, 1-tm-12.2*dh, 'ctrl+h or cmd+h: Open the help box figure.', ha='left', va='top', size=12)
                self.h_fig.show()
        elif event.key == 'm':
            self.t = int(round(event.xdata/self.dt))
            self.chTime()
            self.r = int(round(event.ydata/self.dl))
            self.chDistance()
        elif event.key == 'cmd+g' or event.key =='ctrl+g':
            v = self.glines[0].get_visible()
            v ^= True
            for gl in self.glines:
                gl.set_visible(v)
            self.fig.canvas.draw_idle()

                

    def _next(self):
        if self.t < self.nt-1:
            self.t += 1
        elif self.t >= self.nt-1:
            self.t = 0
        self.chTime()

    def _prev(self):
        if self.t > 0:
            self.t -= 1
        elif self.t <= 0:
            self.t = self.nt-1
        self.chTime()

    def _down(self):
        if self.r > 0:
            self.r -= 1
        elif self.r <= 0:
            self.r = self.nl-1
        self.chDistance()
        
    def _up(self):
        if self.r < self.nl-1:
            self.r += 1
        elif self.r >= self.nl-1:
            self.r = 0
        self.chDistance()
        
    def chTime(self):
        self.vline.set_xdata([self.t*self.dt, self.t*self.dt])
        self.vline_sub.set_xdata([self.t*self.dt, self.t*self.dt])
        self.DS.set_xdata(self.TD[:,self.t])
        if self.parent is not None:
            self.parent.t = self.t
            self.parent.im.set_data(self.parent.data[self.t])
            self.parent.ax.set_title(f't = {self.parent.t*self.dt} (sec)')
            self.parent.fig.canvas.draw_idle()
        self.hline_sub.set_xdata(self.axD.get_xlim())

        self.fig.canvas.draw_idle()

    def chDistance(self):
        self.hline.set_ydata([self.r*self.dl, self.r*self.dl])
        self.hline_sub.set_ydata([self.r*self.dl, self.r*self.dl])
        self.TS.set_ydata(self.TD[self.r])
        self.vline_sub.set_ydata(self.axT.get_ylim())
        self.fig.canvas.draw_idle()

    def save(self, fname=None):
        """
        """
        if fname is None:
            fname = join(getcwd(), f"{self.label}.npz")
        if fname.split('.')[-1] != 'npz':
            raise ValueError("File extension should be npz.")
        np.savez(fname, TD=self.TD, dl=self.dl, dt=self.dt, vposition=self.spGrad, velocity=self.sv, boolFit=self.sfit, period=self.Tperiod, pposition=self.Tline_pos, wavelength=self.Ddistance, wposition=self.Dline_pos)