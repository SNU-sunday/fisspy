import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .. import cm
from ..preprocess.proc_base import fname2isot
from ..read import FISS
from astropy.time import Time
from os.path import join, dirname, basename, isdir, isfile
from os import mkdir
from shutil import move
from glob import glob
from zipfile import ZipFile

__author__ = "Juhyung Kang"
__all__ = ['makeRasterSet']

class makeRasterSet:
    """
    Make Raster image set

    Parameters
    ----------
    flistA: `list`
        list of the cam A file (one among proc, comp data)
    flistB: `list`
        list of the cam B file (one among proc, comp data)
    wvset: `~numpy.ndarray` (optional)
        1D-array for relative wavelength set to draw raster image.
        default is [-4, -0.7, -0.5, -0.2, 0, 0.2, 0.5, 0.7]
    ii: `int` (optional)
        time index to show initially
        default is 0
    show: `bool` (optional)
        show plot
        default is True
        Please set this value to False to save the image or animation.
    
    Other parameters
    ----------------
    **kwargs:
        `~fisspy.read.FISS` keyword arguments.
    """
    def __init__(self, flistA, flistB, wvset=None, ii=None, show=True, **kwargs):

        
        self.show = show
        if show:
            plt.ion()
        else:
            plt.ioff()
        flistA.sort()
        flistB.sort()
        self.flistA = flistA
        self.flistB = flistB
        self.ani = None
        self.nf = len(self.flistA)
        self.kwg = kwargs
        self.fname_movie = None
        bgcolor = "#212529"
        bg_second = "#484c4f"
        fontcolor = "#adb5bd"
        titlecolor = "#ffda6a"
        self.time = np.zeros(self.nf, dtype=float)
        self.anx = np.zeros(self.nf, dtype=int)

        if ii is None:
            idx = self.nf//2

        A, B, time = self.loadData(idx)
        self.rh = A.header
        cwvA = A.centralWavelength
        cwvB = B.centralWavelength

        if wvset is None:
            wvSet = np.array([-4, -0.7, -0.5, -0.2, 0, 0.2, 0.5, 0.7])
        else:
            wvSet = wvset

        self.stT = Time(fname2isot(self.flistA[0]))
        self.stJD = self.stT.jd
        self.edT = Time(fname2isot(self.flistA[-1]))
        self.edJD = self.edT.jd
        self.dJD = self.edJD-self.stJD
        
        self.nwv = nwv = len(wvSet)
        

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
        
        self.nx = A.nx
        self.ny = A.ny
        self.nw = A.nwv
        self.title = self.tax.text(0.5,0.5, time, transform=self.tax.transAxes, ha='center', va='center', weight='bold', size=15, c=titlecolor)
        self.status.set_extent([0, (time.jd-self.stJD)/self.dJD, 0, 1])
        self.imRasterA = [None]*self.nwv
        self.imRasterB = [None]*self.nwv
        self.fig.set_figwidth(self.figy/2.3*self.nx/self.ny*self.nwv)

        self.wvA = cwvA+wvSet
        self.wvB = cwvB+wvSet

        for i in range(nwv):
            self.ax[0, i].set_position([i/nwv,0,1/nwv,0.1/2.31])
            self.ax[1, i].set_position([i/nwv,0.1/2.31,1/nwv,1/2.31])
            self.ax[2, i].set_position([i/nwv,1.1/2.31,1/nwv,0.1/2.31])
            self.ax[3, i].set_position([i/nwv,1.2/2.31,1/nwv,1/2.31])
            if i == nwv//2:
                self.ax[0, i].text(0.5, 0.5, f'{cwvB:.1f} $\\AA$', transform=self.ax[0, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.ax[2, i].text(0.5, 0.5, f'{cwvA:.1f} $\\AA$', transform=self.ax[2, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
            else:
                self.ax[0, i].text(0.5, 0.5, f'{wvSet[i]:.1f} $\\AA$', transform=self.ax[0, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.ax[2, i].text(0.5, 0.5, f'{wvSet[i]:.1f} $\\AA$', transform=self.ax[2, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)

            aa = A.getRaster(self.wvA[i])
            M = aa.max()
            if M > 1e2:
                m = aa[aa>1e2].min()
            else:
                m = aa.min()
            aa = np.log10(aa)
            m = np.log10(m)
            M = np.log10(M)
            self.imRasterA[i] = self.ax[3, i].imshow(aa, cm.ha, origin='lower', clim=[m, M])
            bb = B.getRaster(self.wvB[i])
            M = bb.max()
            if M > 1e2:
                m = bb[bb>1e2].min()
            else:
                m = bb.min()
            bb = np.log10(bb)
            m = np.log10(m)
            M = np.log10(M)
            self.imRasterB[i] = self.ax[1, i].imshow(bb, cm.ca, origin='lower', clim=[m, M])
            for j in range(4):
                self.ax[j, i].set_axis_off()
                self.ax[j, i].set_facecolor(bgcolor)
        if show:
            self.fig.show()

    def loadData(self, i):
        """
        Load Data
        
        Parameters
        ----------
        i: `int`
            Frame Number
            
        Returns
        -------
        A: `~fisspy.read.FISS`
            FISS output for cam A
        B: `~fisspy.read.FISS`
            FISS output for cam B
        time: astropy.time
            Time in isot.
        """
        A = FISS(self.flistA[i], **self.kwg)
        B = FISS(self.flistB[i], **self.kwg)
        time = Time(A.date)

        return A, B, time

    def chData(self, i):
        """
        Change Data shown in figure
        
        Parameters
        ----------
        i: `int`
            Frame Number

        Returns
        -------
        None
        """
        A, B, time = self.loadData(i)
        nx = A.nx
        self.time[i] = time.jd
        self.anx[i] = nx
        self.status.set_extent([0, (time.jd-self.stJD)/self.dJD, 0, 1])
        self.title.set_text(time.isot)
        if self.nx != nx:
            self.nx = nx
            self.fig.set_figwidth(self.figy/2.3*self.nx/self.ny*self.nwv)
            for i in range(self.nwv):
                self.imRasterA[i].set_extent([-0.5, self.nx-0.5,-0.5, self.ny-0.5])
                self.imRasterB[i].set_extent([-0.5, self.nx-0.5,-0.5, self.ny-10-0.5])


        for i in range(self.nwv):
            self.imRasterA[i].set_data(np.log10(A.getRaster(self.wvA[i])))
            self.imRasterB[i].set_data(np.log10(B.getRaster(self.wvB[i])))

        self.fig.canvas.draw_idle()

    def saveAllImages(self, dirn, dpi=100):
        """
        Save all images
        
        Parameters
        ----------
        dirn: `str`
            Save directory
        dpi: `int`, (optional)
            Dots per inch.
            Default is 100.

        Returns
        -------
        None
        """
        for i in range (self.nf):
            fname = join(dirn, self.title.get_text().replace(':','_').replace('-','_')+'.png')
            self.saveImage(fname, i=i, dpi=dpi)
        
        self.saveAnimation(dirn)

    def saveImage(self, fname, i=None, dpi=100):
        """
        Save image for given frame i.

        Parameters
        ----------
        fname: `str`
            Save filename 
        i: `int`, (optional)
            Frame number
            If None, save current frame.
            Default is None.
        dpi: `int`, (optional)
            Dots per inch.
            Default is 100.

        Returns
        -------
        None
        """
        if i is not None:
            self.chData(i)
        self.fig.savefig(fname, dpi=dpi)
        

    def animation(self, interval=100):
        """
        Make animation and show

        Parameters
        ----------
        interval: `int`, (optional)
            Frame interval in unit of ms.
            Default is 100.

        Returns
        -------
        None
        """
        self.ani = FuncAnimation(self.fig, self.chData, frames=np.arange(self.nf), interval=interval)
        self.fig.canvas.draw_idle()
            

    def saveAnimation(self, dirn, interval=100):
        """
        Save animation
        
        Parameters
        ----------
        dirn: `str`
            Save Directory
        interval: `int`, (optional)
            Frame interval in unit of ms.
            Default is 100.

        Returns
        -------
        None
        """
        if self.ani is None:
            self.animation(interval=interval)
        tmp = self.rh['target'].replace(' ', '')
        mname = join(dirn, tmp+'_01.mp4')
        if isfile(mname):
            lf = len(glob(join(dirn, tmp+'*.mp4')))
            mname = mname.replace("01.mp4", f"{lf+1:02}.mp4")
        self.ani.save(mname)
        self.fname_movie = mname
        if not self.show:
            plt.close(self.fig)

    def makeCatalogFiles(self, dirn, interval=100, incdata=True):
        """
        Make JSON file for the data catalog

        Parameters
        ----------
        dirn: `str`
            Save directory.
        interval: `int`, (optional)
            Frame interval in unit of ms.
            Default is 100.
        incdata: `bool`
            If true include data in the JSON file.
            Default is True.

        Returns
        -------
        None
        """
        if self.fname_movie is None:
            self.saveAnimation(dirn, interval=interval)
        
        bdir = dirn
        date = self.stT.isot[:10].replace('-','')
        mdir = join(bdir, 'movie')
        idir = join(bdir, 'img')
        ddir = join(bdir, 'data')
        if not isdir(mdir):
            mkdir(mdir)
        if not isdir(idir):
            mkdir(idir)
        if not isdir(ddir):
            mkdir(ddir)

        amname = basename(self.fname_movie)
        if isfile(join(mdir,amname)):
            lf = len(glob(join(mdir, amname[:-6]+'*.mp4')))
            amname = amname.replace(amname[-6:], f"{lf+1:02}.mp4")
        move(self.fname_movie, join(mdir, amname))

        # make image
        A, B, time = self.loadData(self.nf//2)
        h = A.header
        ifname = self.title.get_text().replace(':','_').replace('-','_')+'.png'
        self.saveImage(join(idir, ifname), i=self.nf//2)

        # make zip file
        zipname0 = h['target'].replace(' ', '')
        zipname = join(ddir, zipname0+'_01.zip')
        if isfile(zipname):
            lf = len(glob(join(ddir, zipname0+'*.zip')))
            zipname = zipname.replace("01.zip", f"{lf+1:02}.zip")
        zp = ZipFile(zipname, 'w')
        for i in range(len(self.flistA)):
            zp.write(self.flistA[i])
            zp.write(self.flistB[i])
        zp.close()

        # input
        observer = h['observer']
        st = self.stT.isot[11:]
        ed = self.edT.isot[11:]
        obstime = f"{st} ~ {ed}"
        try:
            target = h['target']
        except:
            target = 'None'
        try:
            position = f"""["{h['tel_xpos']}", "{h['tel_ypos']}"]"""
        except:
            position = f"""["", ""]"""
        tt = np.roll(self.time,-1) - self.time
        dt = np.median(tt[:-1])*24*3600
        nx = int(np.median(self.anx))
        ny = self.ny
        ax = nx*0.16
        ay = ny*0.16

        # write json
        fjson = join(bdir, f"01_{date}.json")
        if isfile(fjson):
            k = glob(join(bdir,f'*_{date}.json'))
            nk = len(k)
            fjson = join(bdir, f"{nk+1:02}_{date}.json")

        opn = open(fjson, 'w')
        opn.write('{\n')
        opn.write(f"""  "observer": "{observer}",\n""")
        opn.write(f"""  "obstime": "{obstime}",\n""")
        opn.write(f"""  "target": "{target}",\n""")
        opn.write(f"""  "position": {position},\n""")
        opn.write(f"""  "cadence": "{dt:.2f}",\n""")
        opn.write(f"""  "obsarea": ["{ax:.0f}", "{ay:.0f}"],\n""")
        opn.write(f"""  "imgA": "{ifname}",\n""")
        opn.write(f"""  "imgB": "",\n""")
        opn.write(f"""  "movie": ["{amname}"],\n""")
        if incdata:
            opn.write(f"""  "data": ["{basename(zipname)}"]\n""")
        else:
            opn.write(f"""  "data": [""]\n""")
        opn.write('}')
        opn.close()