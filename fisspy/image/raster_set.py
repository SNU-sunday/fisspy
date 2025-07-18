import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from .. import cm
from ..preprocess.proc_base import fname2isot
from ..read import FISS
from astropy.time import Time
from os.path import join, dirname, basename, isdir, isfile, getsize
from os import mkdir, rename
from shutil import move
from glob import glob
from zipfile import ZipFile
from astropy.io import fits
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import json
import matplotlib.ticker as ticker


__author__ = "Juhyung Kang"
__all__ = ['makeRasterSet', 'makeOBSmovie']

def filesize(f):
    fs = getsize(f)
    size = fs/1024**3
    return f"{size:.2f} GB"

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
    def __init__(self, flistA, flistB, wvset=[-4.0,-0.5,0,0.5], ii=None, show=True, **kwargs):

        
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
        else:
            idx = ii

        A, B, time = self.loadData(idx)
        self.rh = A.header
        cwvA = A.centralWavelength
        cwvB = B.centralWavelength

        wvSet = wvset
        
        if type(wvSet) != np.ndarray:
            wvSet = np.array(wvSet)

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
        time = Time(A.header['strtime'])

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

    def makeCatalogFiles(self, dirn, events=[""], seeing="", pubbridge=[""], coobs=[""], note="", interval=100, incdata=True):
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
        A, B, time = self.loadData(self.idx)
        h = A.header
        ifname = self.title.get_text().replace(':','_').replace('-','_')+'.png'
        self.saveImage(join(idir, ifname), i=self.idx)

        # make zip file
        zipname0 = h['target'].replace(' ', '')
        zipname = join(ddir, zipname0+'_01.zip')
        if isfile(zipname):
            lf = len(glob(join(ddir, zipname0+'*.zip')))
            zipname = zipname.replace("01.zip", f"{lf+1:02}.zip")
        zp = ZipFile(zipname, 'w')
        flist = glob(join(self.compD, '*.fts'))
        flist.sort()
        for f in flist:
            zp.write(f)
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
        etmp = """"""
        for ev in events:
            etmp += f""""{ev}", """ 
        
        opn.write(f"""  "keywords": [{etmp[:-2]}]""")
        opn.write(f"""  "seeing": "{seeing}" """)

        tmp = """"""
        for pb in pubbridge:
            tmp += f""""{pb}", """ 
        opn.write(f"""  "pubbridge": [{tmp[:-2]}]""")

        tmp = """"""
        for co in coobs:
            tmp += f""""{co}", """ 
        opn.write(f"""  "coobs": [{tmp[:-2]}]""")
        opn.write(f"""  "note": "{note}" """)

        opn.write('}')
        opn.close()
 
def flipInv(invD):
    lia = glob(join(invD, "*A1_par.fts"))
    lib = glob(join(invD, "*B1_par.fts"))
    lia.sort()
    lib.sort()
    ninvD = join(invD, 'flip')

    if not isdir(ninvD):
        mkdir(ninvD)
    nf = len(lia)

    for i in range(nf):
        A = fits.open(lia[i])[0]
        B = fits.open(lib[i])[0]

        hdu = fits.PrimaryHDU(A.data[...,::-1], A.header)
        hdu.writeto(join(ninvD, basename(lia[i])), overwrite=True)
        hduB = fits.PrimaryHDU(B.data[...,::-1], B.header)
        hduB.writeto(join(ninvD, basename(lib[i])), overwrite=True)

def pjsonIMGtag(dirn):
    idir = join(join(dirname(dirname(dirn)),'img'), 'pub')
    pjL = glob(join(dirn,'*.json'))

    for f in pjL:
        oj = open(f, 'r')
        js = json.load(oj)
        oj.close()
        keys = list(js.keys())

        biname = basename(js['img'])
        if biname:
            ifile = join(idir, biname)
            ext = ifile.split('.')[-1]
            iname = basename(js['adsurl']).replace("&","") + f'.{ext}'
            js['img'] = iname
            # rename(ifile, join(idir,iname))
        else:
            if js['adsurl']:
                iname = basename(js['adsurl']).replace("&","") + '.png'
                js['img'] = iname
            else:
                js['img'] = ""

        opn = open(f, 'w')
        opn.write("{\n")
        nk = len(keys)
        if 'bridge' in keys:
            js.pop('bridge')
            keys = list(js.keys())
            nk = len(keys)
        for i, k in enumerate(keys):
            if type(js[k]) == str:
                tmp = js[k].replace('"', "'")
                txt = f"""  "{k}": "{tmp}" """
            else:
                txt = f'  "{k}": {js[k]}'.replace("'",'"')
            if i != nk-1:
                txt += ',\n'
            else:
                txt +='\n'
            opn.write(txt)
        opn.write("}")
        opn.close()

def zipComp(dirn, saveD, fjson=None):
    flist = glob(join(dirn, '*.fts'))
    flist.sort()
    if fjson is not None:
        oj = open(fjson, 'r')
        js = json.load(oj)
        oj.close()
        try:
            zname = join(saveD,js['data'])
        except:
            zname = join(saveD,js['data_com'])
    else:
        idx = len(flist)//2
        time = fname2isot(flist[idx])
        namebase = join(saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',""))
        zname = namebase+'_comp.zip'

    zp = ZipFile(zname, 'w')
    for f in flist:
        zp.write(f, arcname=basename(f))
    zp.close()
    print(zname)

def upateJSON(fjson, lparam, lvalue):
    """
    Update the json file revising the parameter values.

    Parameters
    ----------
    fjson : `str`
        The filename of the json file to be updated.
    lparam : `list`
        The list of parameters to be updated.
    lvalue : `list`
        The list of values corresponding to the parameters to be updated.
    """
    oj = open(fjson, 'r')
    js = json.load(oj)
    oj.close()

    for p, v in zip(lparam, lvalue):
        if p in js:
            js[p] = v
        else:
            print(f"Parameter '{p}' not found in the JSON file.")

    saveJSON(fjson, js)

def saveJSON(fjson, json_data):
    """
    Save the JSON data to a file.

    Parameters
    ----------
    fjson : `str`
        The filename of the JSON file to be saved.
    json_data : `dict`
        The JSON data to be saved.
    """
    opn = open(fjson, 'w')
    opn.write("{\n")

    js = json_data
    keys = list(js.keys())
    nk = len(keys)
    for i, k in enumerate(keys):
        if type(js[k]) == str:
            tmp = js[k].replace('"', "'")
            txt = f"""  "{k}": "{tmp}" """
        else:
            txt = f'  "{k}": {js[k]}'.replace("'",'"')
        if i != nk-1:
            txt += ',\n'
        else:
            txt +='\n'
        opn.write(txt)
    opn.write("}")
    opn.close()

class makeOBSmovie:
    def __init__(self, compD, saveD, invD=None, recData=False, target="", events=[""], position=["",""], publication=[""], coobs=[""], note="", **FISSkwargs):
        self.tdur = None
        self.tdurI = None
        self.compD = compD
        self.invD = invD
        self.saveD = saveD
        self.kwg = FISSkwargs
        self.sn_cont = None
        self.sn_raster = None
        self.sn_rasterani = None
        self.tjd = None
        self.json = {"title": "",
                     "date": "",
                     "obstime": "",
                     "duration": [""],
                     "cadence": 0.,
                     "target": target,
                     "events": events,
                     "position": position,
                     "obsarea": ["", ""],
                     "observer": "",
                     "publication": publication,
                     "img_target": "",
                     "img_raster": "",
                     "movie_raster": "",
                     "data_com": "",
                     "data_inv": "",
                     "size_com": "",
                     "size_inv": "",
                     "recommend": int(recData),
                     "coobs": coobs,
                     "note": note}
        
        self.lca =  glob(join(compD, "*A1_c.fts"))
        self.lcb =  glob(join(compD, "*B1_c.fts"))
        if invD is not None:
            self.lia =  glob(join(invD, "*A1_par.fts"))
            self.lib =  glob(join(invD, "*B1_par.fts"))
            self.lia.sort()
            self.lib.sort()
        self.lca.sort()
        self.lcb.sort()
        self.nf = len(self.lca)
        a,b,t0 = self.loadData(0)
        a = FISS(self.lca[0], **self.kwg)
        b = FISS(self.lcb[0], **self.kwg)
        self.t0 = t0
        self.nya, self.nxa, self.nwa = a.data.shape
        self.nyb, self.nxb, self.nwb = b.data.shape
        self.cwva = a.cwv
        self.wva = a.wave - a.cwv
        self.dwa = np.median(np.diff(self.wva))
        self.cwvb = b.cwv
        self.wvb = b.wave - b.cwv
        self.dwb = np.median(np.diff(self.wvb))
        self.binit = False
        self.dsa = None

    def _initDS(self, wvset):
        self.tjd0 = np.zeros(self.nf, dtype=float)
        for i, f in enumerate(self.lca[:self.nf]):
            tstr = fname2isot(f)
            self.tjd0[i] = Time(tstr).jd*24*3600
        self.tjd0 -= self.tjd0[0]
        dt = np.median(np.diff(self.tjd0))
        nt = int(round((self.tjd0[-1] / dt)))
        self.newT = np.arange(nt)*dt
        self.dsa = np.zeros((self.nwa, nt), dtype=float)
        self.dsb = np.zeros((self.nwb, nt), dtype=float)
        self.extA = [self.newT[0]-dt*0.5, self.newT[-1]+dt*0.5, self.wva[0] + self.dwa/2, self.wva[-1] + self.dwa/2]
        self.extB = [self.newT[0]-dt*0.5, self.newT[-1]+dt*0.5, self.wvb[-1] - self.dwb/2, self.wvb[0] - self.dwb/2]
        self.rasterA = np.zeros((self.nf, len(wvset), self.nya, self.nxa))
        self.rasterB = np.zeros((self.nf, len(wvset), self.nyb, self.nxb))
        self.binit = True

    def saveAll(self, ii=None, wvset=[-4,-0.5,0,0.5]):
        if ii is None:
            idx = self.nf//2
        else:
            idx = ii

        self.idx = idx
        fig = self.Itarget(idx, wvset[0], save=True)
        plt.close(fig)
        fig = self.Iraster(idx, wvset, save=True)
        plt.close(fig)
        self.updateJSON()
        self.saveJSON()

    def saveJSON(self):
        time = fname2isot(self.lca[self.idx])
        jname = join(self.saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',"")+".json")
        saveJSON(jname, self.json)
        print(f"Save json file: {jname}")

    def reviseJSON(self, key, arg):
        if key == 'publication' and type(arg) == list:
            for i,p in enumerate(arg):
                arg[i] = basename(p)
        if type(arg) == list:
            self.json[key] = f"{arg}".replace("'",'"')
        self.json[key] = arg

    def updateJSON(self):

        A = FISS(self.lca[self.idx], **self.kwg)
        h = A.header
        observer = h['observer']
        st = Time(fname2isot(self.lca[0])).isot[11:]
        ed = Time(fname2isot(self.lca[-1])).isot[11:]
        obstime = f"{st} ~ {ed}"
        if not self.json["position"][0]:
            try:
                position = f"""["{h['tel_xpos']}", "{h['tel_ypos']}"]"""
            except:
                position = f"""["", ""]"""
        else:
            position = self.json["position"]
        if not self.json["target"]:
            try:
                target = h['target']
            except:
                target = 'None'
        else:
            target = self.json["target"]

        tmp = fname2isot(self.lca[self.idx])
        date = tmp[:10]
        title = f'{date} ({target})'
        nf = len(self.lca)
        if self.tjd is None:
            self.tjd = np.zeros(nf, dtype=float)
            for i, f in enumerate(self.lca):
                tstr = fname2isot(f)
                self.tjd[i] = Time(tstr).jd
        tt = np.roll(self.tjd,-1) - self.tjd
        tsec = tt*3600*24
        cadence = np.median(tsec[:-1])
        wh = tsec >= cadence*1.5
        w0 = 0
        dur = """["""
        wh2 = np.arange(nf)[wh]
        tsec = self.tjd*3600*24
        for w in wh2:
            dt = (tsec[w]-tsec[w0])/60 # in min
            st = Time(fname2isot(self.lca[w0])).isot[11:]
            ed = Time(fname2isot(self.lca[w])).isot[11:]
            dur += f'"{st} ~ {ed} ({dt:.1f} min)", '
            w0 = w+1
        dt = (tsec[-1]-tsec[w0])/60
        st = Time(fname2isot(self.lca[w0])).isot[11:]
        ed = Time(fname2isot(self.lca[-1])).isot[11:]
        dur += f'"{st} ~ {ed} ({dt:.1f} min)"]'

        self.reviseJSON('date', date)
        self.reviseJSON('title', title)
        self.reviseJSON('observer', observer)
        self.reviseJSON('duration', dur)
        self.reviseJSON('obstime', obstime)
        self.reviseJSON('position', position)
        self.reviseJSON('target', target)
        self.reviseJSON('obsarea', f"""["{A.nx*0.16:.0f}", "{A.ny*0.16:.0f}"]""")
        self.reviseJSON('cadence', f"{cadence:.2f}")
        time = fname2isot(self.lca[self.idx])
        sd = join(self.saveD, 'data')
        if not isdir(sd):
            mkdir(sd)
        namebase = join(sd, time[:10].replace('-', '')+'_'+time[11:].replace(':',""))
        if self.sn_raster is not None:
            self.reviseJSON('img_raster', basename(self.sn_raster))
        if self.sn_rasterani is not None:
            self.reviseJSON('movie_raster', basename(self.sn_rasterani))
            czipn = namebase+'_comp.zip'
            flist = glob(join(self.compD, '*.fts'))
            flist.sort()
            if not isfile(czipn):
                zp = ZipFile(czipn, 'w')
                for f in flist:
                    zp.write(f, arcname=basename(f))
                zp.close()
            self.reviseJSON("data_com", basename(czipn))
            self.reviseJSON("size_com", filesize(czipn))
        if self.invD is not None:
            izipn = namebase+'_inv.zip'
            if not isfile(izipn):
                zp = ZipFile(izipn, 'w')
                for i in range(len(self.lia)):
                    zp.write(self.lia[i], arcname=basename(self.lia[i]))
                    zp.write(self.lib[i], arcname=basename(self.lib[i]))
                zp.close()
            self.reviseJSON("data_inv", basename(izipn))
            self.reviseJSON("size_inv", filesize(izipn))
        if self.sn_cont is not None:
            self.reviseJSON('img_target', basename(self.sn_cont))

        if type(self.json['events']) == list:
            self.json['events'] = f"{self.json['events']}".replace("'",'"')
        if type(self.json['position']) == list:
            self.json['position'] = f"{self.json['position']}".replace("'",'"')
        if type(self.json['publication']) == list:
            self.json['pubname'] = [None]*len(self.json['publication'])
            for i,p in enumerate(self.json['publication']):
                self.json['publication'][i] = basename(p)
                url = 'https://ui.adsabs.harvard.edu/abs/' + self.json['publication'][i]
                name = self.getPub(url)
                self.json['pubname'][i] = name
            self.json['publication'] = f"{self.json['publication']}".replace("'",'"')
            self.json['pubname'] = f"{self.json['pubname']}".replace("'",'"')
        if type(self.json['coobs']) == list:
            self.json['coobs'] = f"{self.json['coobs']}".replace("'",'"')

    def getPub(self, url):
        opn = urlopen(url+'/abstract')
        par = bs(opn.read(), 'html.parser')
        meta = par.find_all('meta')
        opn.close()

        for m in meta:
            if m.get('name') == 'citation_authors':
                authors =  m.get('content')

        na = authors.count(';')
        name = authors.split(',')[0]
        if na == 1:
            name += f" ({basename(url)[:4]})"
        else:
            name += f" et al. ({basename(url)[:4]})"
        return name

    def Itarget(self, ii=None, wv=-4, save=True):
        if ii is None:
            idx = self.nf//2
        else:
            idx = ii
        self.idx = idx
        A, B, time = self.loadData(idx)
        cwv = A.centralWavelength
        nx = A.nx
        ny = A.ny
        figy = 7
        r = nx/ny
        fig, ax = plt.subplots(figsize=[r*figy,figy])
        ax.set_position([0,0,1,1])
        aa = A.getRaster(cwv+wv)
        M = aa.max()
        if M > 1e2:
            m = aa[aa>1e2].min()
        else:
            m = aa.min()
        aa = np.log10(aa)
        m = np.log10(m)
        M = np.log10(M)
        ax.imshow(aa, cm.ha, origin='lower', clim=[m,M])
        fig.show()
        if save:
            sd = join(self.saveD, 'img')
            if not isdir(sd):
                mkdir(sd)
            time = fname2isot(self.lca[self.idx])
            tmp = join(sd, time[:10].replace('-', '')+'_'+time[11:].replace(':',"")+"_cont.png")
            print(tmp)
            self.sn_cont = tmp
            print(f"Save continuum figure: {self.sn_cont}")
            fig.savefig(self.sn_cont)
            return fig
            # self.updateJSON()

    def loadData(self, i):
        A = FISS(self.lca[i], **self.kwg)
        B = FISS(self.lcb[i], **self.kwg)
        time = Time(fname2isot(self.lca[i]))

        return A, B, time

    def makeData(self, wvset=[-4,-0.5,0,0.5]):
        self.wvset = wvset
        self._initDS(wvset)
        # wvseta = np.array(wvset) + self.cwva
        # wvsetb = np.array(wvset) + self.cwvb
        
        self.isot = [None]*self.nf
        self.nwvset = len(wvset)
        for i in range(self.nf):
            A, B, time = self.loadData(i)
            self.isot[i] = time
            it = np.abs(self.newT-self.tjd0[i]).argmin()
            self.dsa[:, it] = A.data.mean((0,1))
            self.dsb[:, it] = B.data.mean((0,1)).T

            for j,w in enumerate(wvset):
                self.rasterA[i,j] = A.getRaster(self.cwva + w)
                self.rasterB[i,j] = B.getRaster(self.cwvb + w)
        wh = (self.dsa[200] > 0)
        self.dsa -= self.dsa[:,wh].mean(1)[:, None]
        self.dsb -= self.dsb[:,wh].mean(1)[:, None]
        self.dsa[:, ~wh] = 0
        self.dsb[:, ~wh] = 0

    def Iraster(self, ii=None, wvset=[-4,-0.5,0,0.5], save=False, interval=100):
        bgcolor = "#212529"
        bg_second = "#484c4f"
        fontcolor = "#adb5bd"
        titlecolor = "#ffda6a"

        if self.dsa is None:
            self.makeData(wvset)
        
        if ii is None:
            idx = self.nf//2
        else:
            idx = ii

        t = self.isot[idx]
        
        nx = self.nxa
        ny = self.nya
        nwv = self.nwvset

        fy = 7
        fx = 19
        dy = ny/6
        ty = 3*dy+2*ny
        trx = nx*nwv
        xds = fx/fy*ty-trx-2.5*dy
        tx = trx + xds+2.5*dy

        wr = nx/tx
        hr = ny/ty
        dr = dy/ty
        drx = dy/tx*2
        wds = xds/tx

        fs = [fx, fy]

        self.fig, self.ax = plt.subplots(4, nwv, figsize=fs)
        self.fig.set_facecolor(bgcolor)
        self.axH = self.ax[0]
        self.axHt = self.ax[1]
        self.axC = self.ax[2]
        self.axCt = self.ax[3]
        self.axTitle = self.fig.add_subplot(121)
        self.axTitle.set_position([0, (hr+dr)*2, nwv*wr, dr])

        self.axDSH = self.fig.add_subplot(121)
        self.axDSH.set_position([nwv*wr+drx, hr+dr*2, wds, hr])
        self.axDSH.set_facecolor(bgcolor)

        self.axDSC = self.fig.add_subplot(121)
        self.axDSC.set_position([nwv*wr+drx, dr, wds, hr])
        self.axDSC.set_facecolor(bgcolor)

        self.axTitle.set_axis_off()

        self.imRA = [None]*nwv
        self.imRB = [None]*nwv

        for i in range(nwv):
            self.axCt[i].set_facecolor(bgcolor)
            self.axC[i].set_facecolor(bgcolor)
            self.axHt[i].set_facecolor(bgcolor)
            self.axH[i].set_facecolor(bgcolor)
            self.axCt[i].set_position([i*wr, 0, wr, dr])
            self.axC[i].set_position([i*wr, dr, wr, hr])
            self.axHt[i].set_position([i*wr, hr+dr, wr, dr])
            self.axH[i].set_position([i*wr, hr+dr*2, wr, hr])
            self.axCt[i].set_axis_off()
            self.axC[i].set_axis_off()
            self.axHt[i].set_axis_off()
            self.axH[i].set_axis_off()

            aa = self.rasterA[idx, i]
            M = aa.max()
            if M > 1e2:
                m = aa[aa>1e2].min()
            else:
                m = aa.min()
            aa = np.log10(aa)
            m = np.log10(m)
            M = np.log10(M)
            self.imRA[i] = self.axH[i].imshow(aa, cm.ha, origin='lower', clim=[m, M])

            bb = self.rasterB[idx, i]
            M = bb.max()
            if M > 1e2:
                m = bb[bb>1e2].min()
            else:
                m = bb.min()
            bb = np.log10(bb)
            m = np.log10(m)
            M = np.log10(M)
            self.imRB[i] = self.axC[i].imshow(bb, cm.ca, origin='lower', clim=[m, M])

            if i == nwv//2:
                self.axHt[i].text(0.5, 0.5, f'{self.cwva:.1f} $\\AA$', transform=self.axHt[i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.axCt[i].text(0.5, 0.5, f'{self.cwvb:.1f} $\\AA$', transform=self.axCt[i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
            else:
                self.axHt[i].text(0.5, 0.5, f'{self.wvset[i]:.1f} $\\AA$', transform=self.axHt[i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.axCt[i].text(0.5, 0.5, f'{self.wvset[i]:.1f} $\\AA$', transform=self.axCt[i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)

        self.titleR = self.axTitle.text(0.5,0.4, self.isot[idx].value, transform=self.axTitle.transAxes, ha='center', va='center', weight='bold', size=15, c=titlecolor)
        M = max(np.abs(self.dsa.min()), np.abs(self.dsa.max()))
        self.imDSH = self.axDSH.imshow(self.dsa, plt.cm.RdBu_r, origin='lower', aspect='auto', clim=[-M,M], extent=self.extA, interpolation='nearest')
        M = max(np.abs(self.dsb.min()), np.abs(self.dsb.max()))
        self.imDSC = self.axDSC.imshow(self.dsb, plt.cm.RdBu_r, origin='lower', aspect='auto', clim=[-M,M], extent=self.extB, interpolation='nearest')

        self.axDSH.set_title('Dynamic Spectrum for cam A', color=fontcolor)
        self.axDSC.set_title('Dynamic Spectrum for cam B', color=fontcolor)
        self.axDSC.set_xlabel('Time (s)', color=fontcolor)
        self.axDSH.set_ylabel('Wavelength ($\\AA$)', color=fontcolor)
        self.axDSC.set_ylabel('Wavelength ($\\AA$)', color=fontcolor)
        self.axDSH.tick_params(colors=fontcolor)
        self.axDSC.tick_params(colors=fontcolor)

        self.tlineH = self.axDSH.plot([self.tjd0[idx], self.tjd0[idx]], [self.extA[2], self.extA[3]], ls='dashed', color='k')[0]
        self.tlineC = self.axDSC.plot([self.tjd0[idx], self.tjd0[idx]], [self.extB[2], self.extB[3]], ls='dashed', color='k')[0]

        self.fig.show()

        if save:
            tt = self.isot[idx].value
            sd = join(self.saveD, 'img')
            self.sn_raster = join(sd, tt[:10].replace('-', '')+'_'+tt[11:].replace(':',"")+"_raster.png")
            if not isdir(sd):
                mkdir(sd)
            print(f"Save raster figure: {self.sn_raster}")
            self.fig.savefig(self.sn_raster)
        self.animation(interval, save=save)
        return self.fig

    def chData(self, i):
        self.titleR.set_text(self.isot[i].value)
        self.tlineH.set_xdata([self.tjd0[i], self.tjd0[i]])
        self.tlineC.set_xdata([self.tjd0[i], self.tjd0[i]])

        for j in range(self.nwvset):
            self.imRA[j].set_data(np.log10(self.rasterA[i, j]))
            self.imRB[j].set_data(np.log10(self.rasterB[i, j]))

        self.fig.canvas.draw_idle()

    def animation(self, interval=100, save=True):
        self.ani = FuncAnimation(self.fig, self.chData, frames=np.arange(self.nf), interval=interval)
        self.fig.canvas.draw_idle()
        if save:
            sd = join(self.saveD, 'movie')
            if not isdir(sd):
                mkdir(sd)
            self.sn_rasterani = join(sd, basename(self.sn_raster[:-3])+'mp4')
            if not isdir(sd):
                mkdir(sd)
            print(f"Save raster animation: {self.sn_rasterani}")
            self.ani.save(self.sn_rasterani)
            # self.updateJSON()


def demo():
    cdir = '/Users/jhkang/data/FISS/2024/08/15'
    idir = '/Users/jhkang/data/FISS/2024/08/15/inv'
    sdir = '/Users/jhkang/data/FISS/video_test/240815'
    events = ['transverse MHD waves', 'fibrils']
    target = "Quiet Sun - Part2"
    pos = ["20","-75"]
    pub = ["https://ui.adsabs.harvard.edu/abs/2021JKAS...54..139C", "https://ui.adsabs.harvard.edu/abs/2023ApJ...958..131K"]
    cobs = ["https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20200730_155928_3600011659_2020-07-30T15%3A59%3A282020-07-30T15%3A59%3A28.xml"]
    k = makeOBSmovie(cdir, sdir, idir, events=events, position=pos, publication=pub, target=target, coobs=cobs, recData=False, note="This is a test data for FISS movie making.")
    k.nf = 20
    k.saveAll()


