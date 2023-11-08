import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fisspy.read.readbase import getHeader, readFrame
from fisspy import cm
from astropy.io import fits
from fisspy.preprocess import proc_base
from astropy.time import Time
from os.path import join, basename

class makeRasterSet:
    def __init__(self, flistA, flistB, flatA, flatB, wvset=None, ii=0):
        flistA.sort()
        flistB.sort()
        self.flistA = flistA
        self.flistB = flistB
        self.ani = None
        self.nf = len(self.flistA)
        bgcolor = "#212529"
        bg_second = "#484c4f"
        fontcolor = "#adb5bd"
        titlecolor = "#ffda6a"

        if flistA[ii].find('1.fts')>=0:
            self.level = 1
            h = getHeader(flistA[ii])
            self.dwvA = h['cdelt1']
            self.dwvB = getHeader(flistB[ii])['cdelt1']
            try:
                band = int(h['wavelen'])
            except:
                band = int(h['gratwvln'])
            
        elif flistA[ii].find('_c.fts')>=0:
            self.level = 2
            h = getHeader(flistA[ii])
            self.dwvA = h['cdelt1']
            self.dwvB = getHeader(flistB[ii])['cdelt1']
            try:
                band = int(h['wavelen'])
            except:
                band = int(h['gratwvln'])
        else:
            self.level = 0
            self.dwvA = 0.019
            self.dwvB = -0.026
            h = fits.getheader(flistA[ii])
            try:
                band = int(h['wavelen'])
            except:
                band = int(h['gratwvln'])


        if band == 6562:
            cwvA = 6562.8
            cwvB = 8542.1
        elif band == 5434:
            cwvA = 5434.0
            cwvB = 5883.0

        if wvset is None:
            wvset = np.array([-4, -0.7, -0.5, -0.2, 0, 0.2, 0.5, 0.7])

        self.stJD = Time(proc_base.fname2isot(self.flistA[0])).jd
        self.edJD = Time(proc_base.fname2isot(self.flistA[-1])).jd
        self.dJD = self.edJD-self.stJD
        
        self.nwv = nwv = len(wvset)
        rprofA = flatA[5:-5,5:-5].mean(0)
        wcA = rprofA.argmin()+5
        rprofB = flatB[5:-5,5:-5].mean(0)
        wcB = rprofB.argmin()+5
        self.figy = 8
        self.fig, self.ax = plt.subplots(4,nwv, figsize=[12,self.figy],dpi=100)
        self.fig.set_facecolor(bgcolor)
        self.sax = self.fig.add_subplot(121)
        self.sax.set_position([0,2.2/2.31,1,0.01/2.31])
        self.sax.set_facecolor(bg_second)
        self.sax.set_xlim(0,1)
        self.sax.set_ylim(0,1)
        self.sax.tick_params(left=False, bottom=False,labelleft=False, labelbottom=False)
        # self.status = self.sax.fill_between([0, 0.2],0,1, color='r', alpha=0.7)
        x = np.ones([1,1])
        self.status = self.sax.imshow(x, cmap=plt.cm.hsv, extent=[0,0,0,1], alpha=0.7)
        self.sax.set_aspect(aspect='auto')
        self.tax = self.fig.add_subplot(121)
        self.tax.set_position([0,2.21/2.31,1,0.1/2.31])
        self.tax.set_facecolor(bgcolor)
        self.tax.set_axis_off()
        A, B, time = self.loadData(ii)
        self.ny, self.nx, self.nw = A.shape
        self.title = self.tax.text(0.5,0.5, time, transform=self.tax.transAxes, ha='center', va='center', weight='bold', size=15, c=titlecolor)
        self.status.set_extent([0, (Time(time).jd-self.stJD)/self.dJD, 0, 1])
        self.imRasterA = [None]*self.nwv
        self.imRasterB = [None]*self.nwv
        self.fig.set_figwidth(self.figy/2.3*self.nx/self.ny*self.nwv)

        self.wvpixA = wcA+(wvset/self.dwvA).astype(int)
        self.wvpixB = wcB+(wvset/self.dwvB).astype(int)

        for i in range(nwv):
            self.ax[0, i].set_position([i/nwv,0,1/nwv,0.1/2.31])
            self.ax[1, i].set_position([i/nwv,0.1/2.31,1/nwv,1/2.31])
            self.ax[2, i].set_position([i/nwv,1.1/2.31,1/nwv,0.1/2.31])
            self.ax[3, i].set_position([i/nwv,1.2/2.31,1/nwv,1/2.31])
            if i == nwv//2:
                self.ax[0, i].text(0.5, 0.5, f'{cwvB:.1f} $\\AA$', transform=self.ax[0, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.ax[2, i].text(0.5, 0.5, f'{cwvA:.1f} $\\AA$', transform=self.ax[2, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
            else:
                self.ax[0, i].text(0.5, 0.5, f'{wvset[i]:.1f} $\\AA$', transform=self.ax[0, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.ax[2, i].text(0.5, 0.5, f'{wvset[i]:.1f} $\\AA$', transform=self.ax[2, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)

            self.imRasterA[i] = self.ax[3, i].imshow(A[:, :, self.wvpixA[i]], cm.ha, origin='lower')
            self.imRasterB[i] = self.ax[1, i].imshow(B[:, :, self.wvpixB[i]], cm.ca, origin='lower')
            for j in range(4):
                self.ax[j, i].set_axis_off()
                self.ax[j, i].set_facecolor(bgcolor)

    def loadData(self, i):
        time = proc_base.fname2isot(self.flistA[i])
        if self.level == 0:
            A = fits.getdata(self.flistA[i]).transpose((1,0,2))
            B = fits.getdata(self.flistB[i]).transpose((1,0,2))
        else:
            h = getHeader(self.flistA[i])
            pfile = h.pop('pfile', False)
            A = readFrame(self.flistA[i], pfile)
            h = getHeader(self.flistB[i])
            pfile = h.pop('pfile', False)
            B = readFrame(self.flistB[i], pfile)
        return A, B, time

    def chData(self, i):
        A, B, time = self.loadData(i)
        nx = A.shape[1]
        self.status.set_extent([0, (Time(time).jd-self.stJD)/self.dJD, 0, 1])
        self.title.set_text(time)
        if self.nx != nx:
            self.nx = nx
            self.fig.set_figwidth(self.figy/2.3*self.nx/self.ny*self.nwv)
            for i in range(self.nwv):
                self.imRasterA[i].set_extent([-0.5, self.nx-0.5,-0.5, self.ny-0.5])
                self.imRasterB[i].set_extent([-0.5, self.nx-0.5,-0.5, self.ny-10-0.5])


        for i in range(self.nwv):
            self.imRasterA[i].set_data(A[:, :, self.wvpixA[i]])
            self.imRasterB[i].set_data(B[:, :, self.wvpixB[i]])

        self.fig.canvas.draw_idle()

    def saveAll(self, dirn, stype='png',dpi=100):
        for i in range (self.nf):
            self.chData(i)
            fname = join(dirn,self.title.get_text().replace(':','_').replace('-','_')+f'.{stype}')
            self.fig.savefig(fname, dpi=dpi)

    def animation(self):
        self.ani = FuncAnimation(self.fig, self.chData, frames=np.arange(self.nf), interval=100)
        self.fig.canvas.draw_idle()

    def saveAnimation(self, dirn, stype='mp4'):
        if self.ani is None:
            self.animation()
        time = proc_base.fname2isot(self.flistA[0])
        fname = join(dirn, time[:10] + '.mp4')
        self.ani.save(fname)