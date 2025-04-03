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
__all__ = ['makeRasterSet']

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
    
class makeRecDataJSON:
    def __init__(self, compdir, invdir, savedir, events=[""], position=["",""], publication=[""], coobs=[""], target="", note="", **kwargs):
        self.tdur = None
        self.tdurI = None
        self.compD = compdir
        self.invD = invdir
        self.saveD = savedir
        self.kwg = kwargs
        self.sn_cont = None
        self.sn_raster = None
        self.sn_rasterani = None
        self.sn_inversion = None
        self.sn_inversionani = None
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
                     "img_inv": "",
                     "movie_raster": "",
                     "movie_inv": "",
                     "data_com": "",
                     "data_inv": "",
                     "size_com": "",
                     "size_inv": "",
                     "coobs": coobs,
                     "note": note}
        
        self.lca =  glob(join(compdir, "*A1_c.fts"))
        self.lcb =  glob(join(compdir, "*B1_c.fts"))
        self.lia =  glob(join(invdir, "*A1_par.fts"))
        self.lib =  glob(join(invdir, "*B1_par.fts"))
        self.lca.sort()
        self.lcb.sort()
        self.lia.sort()
        self.lib.sort()
        self.nf = len(self.lca)

    def saveAll(self, ii=None, wvset=[-4,-0.5,0,0.5], excinv=False):
        if ii is None:
            idx = self.nf//2
        else:
            idx = ii

        self.idx = idx
        fig = self.Itarget(idx, wvset[0], save=True)
        plt.close(fig)
        fig = self.Iraster(idx, wvset, save=True)
        plt.close(fig)
        if not excinv:
            fig = self.Iinversion(idx, save=True)
            plt.close(fig)
        self.updateJSON()
        self.saveJSON()

    def saveJSON(self):
        time = fname2isot(self.lca[self.idx])
        jname = join(self.saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',"")+".json")
        opn = open(jname, 'w')
        opn.write('{\n')
        opn.write(f"""  "title": "{self.json['title']}",\n""")
        opn.write(f"""  "date": "{self.json['date']}",\n""")
        opn.write(f"""  "obstime": "{self.json['obstime']}",\n""")
        opn.write(f"""  "duration": {self.json['duration']},\n""")
        opn.write(f"""  "cadence": "{self.json['cadence']}",\n""")
        opn.write(f"""  "target": "{self.json['target']}",\n""")
        opn.write(f"""  "events": {self.json['events']},\n""")
        opn.write(f"""  "position": {self.json['position']},\n""")
        opn.write(f"""  "obsarea": {self.json['obsarea']},\n""")
        opn.write(f"""  "observer": "{self.json['observer']}",\n""")
        opn.write(f"""  "publication": {self.json['publication']},\n""")
        opn.write(f"""  "pubname": {self.json['pubname']},\n""")
        opn.write(f"""  "img_target": "{self.json['img_target']}",\n""")
        opn.write(f"""  "img_raster": "{self.json['img_raster']}",\n""")
        opn.write(f"""  "img_inv": "{self.json['img_inv']}",\n""")
        opn.write(f"""  "movie_raster": "{self.json['movie_raster']}",\n""")
        opn.write(f"""  "movie_inv": "{self.json['movie_inv']}",\n""")
        opn.write(f"""  "data_com": "{self.json['data_com']}",\n""")
        opn.write(f"""  "data_inv": "{self.json['data_inv']}",\n""")
        opn.write(f"""  "size_com": "{self.json['size_com']}",\n""")
        opn.write(f"""  "size_inv": "{self.json['size_inv']}",\n""")
        opn.write(f"""  "coobs": {self.json['coobs']},\n""")
        opn.write(f"""  "note": "{self.json['note']}"\n""")
        opn.write('}')
        opn.close()

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
        if self.tjd is not None:
            tt = np.roll(self.tjd,-1) - self.tjd
            tsec = tt*3600*24
            cadence = np.median(tsec[:-1])
            wh = tsec >= cadence*1.5
            w0 = 0
            dur = """["""
            wh2 = np.arange(self.nf)[wh]
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
        namebase = join(self.saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',""))
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
        if self.sn_inversion is not None:
            self.reviseJSON('img_inv', basename(self.sn_inversion))
        if self.sn_inversionani is not None:
            self.reviseJSON('movie_inv', basename(self.sn_inversionani))
            izipn = namebase+'_inv.zip'
            if not isfile(izipn):
                zp = ZipFile(izipn, 'w')
                for i in range(self.nf):
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

    def Iduration(self, ax):
        I = np.zeros(len(self.lca), dtype=float)
        t = [None]*len(self.lca)
        self.tjd = tjd = np.zeros(len(self.lca), dtype=float)
        a = FISS(self.lca[0], **self.kwg)

        for i, f in enumerate(self.lca):
            a = FISS(f, **self.kwg)
            cpix = abs(a.wave-a.cwv).argmin()
            I[i] = a.data[:,:-1,cpix-50:cpix+50].sum()
            tmp = a.header['strtime'][:10].replace('.', '-')+'T'+a.header['strtime'][11:]
            t[i] = Time(tmp).datetime
            tjd[i] = Time(tmp).jd
        
        ax.plot(t, I/I[0], 'w+-')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_title('Mean Intensity at line center (normalized by the first frame value)', color='w')
        ax.tick_params(axis='both', colors='white')
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        return t

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
            time = fname2isot(self.lca[self.idx])
            tmp = join(self.saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',"")+"_cont.png")
            self.sn_cont = join(self.saveD, tmp)
            print(f"Save continuum figure: {self.sn_cont}")
            fig.savefig(self.sn_cont)
            return fig
            # self.updateJSON()

    def loadData(self, i):
        A = FISS(self.lca[i], **self.kwg)
        B = FISS(self.lcb[i], **self.kwg)
        time = Time(fname2isot(self.lca[self.idx]))

        return A, B, time

    def Iraster(self, ii=None, wvset=[-4,-0.5,0,0.5], save=True, interval=100):
        bgcolor = "#212529"
        bg_second = "#484c4f"
        fontcolor = "#adb5bd"
        titlecolor = "#ffda6a"

        if ii is None:
            idx = self.nf//2
        else:
            idx = ii

        self.idx = idx
        A, B, time = self.loadData(idx)
        self.rh = A.header
        cwvA = A.centralWavelength
        cwvB = B.centralWavelength

        self.nx = A.nx
        self.nx0 = A.nx
        self.ny = A.ny
        self.nw = A.nwv

        if type(wvset) == np.ndarray:
            self.wvset = wvset
        else:
            self.wvset = np.array(wvset)

        self.nwv = nwv = len(self.wvset)

        self.wvA = cwvA+self.wvset
        self.wvB = cwvB+self.wvset

        self.figy = 8
        fs = [self.figy/2.78*self.nx/self.ny*self.nwv, self.figy]
        self.fig, self.ax = plt.subplots(4, nwv, figsize=fs)
        self.fig.set_facecolor(bgcolor)

        self.tax = self.fig.add_subplot(121)
        self.tax.set_position([0,2.72/2.81,1,0.1/2.81])
        self.tax.set_facecolor(bgcolor)
        self.tax.set_axis_off()

        self.axD = self.fig.add_subplot(121)
        self.axD.set_position([0.06,0.03,0.92,0.3/2.81])
        self.axD.set_facecolor(bgcolor)
        if self.tdur is None:
            self.tdur = self.Iduration(self.axD)
        
        time = fname2isot(self.lca[self.idx])
        self.sn_raster = join(self.saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',"")+"_raster.png")

        yl = self.axD.get_ylim()
        self.axD.set_ylim(yl)
        self.pt = self.axD.plot([self.tdur[self.idx], self.tdur[self.idx]], yl, color='r')[0]

        self.titleR = self.tax.text(0.5,0.5, time, transform=self.tax.transAxes, ha='center', va='center', weight='bold', size=15, c=titlecolor)
        self.imRasterA = [None]*nwv
        self.imRasterB = [None]*nwv

        for i in range(nwv):
            self.ax[0, i].set_position([i/nwv,0.48/2.81,1/nwv,0.1/2.81])
            self.ax[1, i].set_position([i/nwv,0.58/2.81,1/nwv,1.02/2.81])
            self.ax[2, i].set_position([i/nwv,1.60/2.81,1/nwv,0.1/2.81])
            self.ax[3, i].set_position([i/nwv,1.70/2.81,1/nwv,1.02/2.81])
            
            if i == nwv//2:
                self.ax[0, i].text(0.5, 0.5, f'{cwvB:.1f} $\\AA$', transform=self.ax[0, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.ax[2, i].text(0.5, 0.5, f'{cwvA:.1f} $\\AA$', transform=self.ax[2, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
            else:
                self.ax[0, i].text(0.5, 0.5, f'{self.wvset[i]:.1f} $\\AA$', transform=self.ax[0, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)
                self.ax[2, i].text(0.5, 0.5, f'{self.wvset[i]:.1f} $\\AA$', transform=self.ax[2, i].transAxes, ha='center', va='center', weight='bold', size=12, c=fontcolor)

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

        self.fig.show()
        if save:
            print(f"Save raster figure: {self.sn_raster}")
            self.fig.savefig(self.sn_raster)
        self.animation(interval, save=save)
        return self.fig

    def chData(self, i):
        A, B, time = self.loadData(i)
        nx = A.nx
        time = fname2isot(self.lca[i])
        self.titleR.set_text(time)
        self.pt.set_xdata([self.tdur[i], self.tdur[i]])
        if self.nx != nx:
            self.nx = nx
            self.fig.set_figwidth(self.figy/2.78*self.nx/self.ny*self.nwv)
            for i in range(self.nwv):
                self.imRasterA[i].set_extent([-0.5, self.nx-0.5,-0.5, self.ny-0.5])
                self.imRasterB[i].set_extent([-0.5, self.nx-0.5,-0.5, self.ny-10-0.5])


        for i in range(self.nwv):
            self.imRasterA[i].set_data(np.log10(A.getRaster(self.wvA[i])))
            self.imRasterB[i].set_data(np.log10(B.getRaster(self.wvB[i])))

        self.fig.canvas.draw_idle()

    def animation(self, interval=100, save=True):
        self.ani = FuncAnimation(self.fig, self.chData, frames=np.arange(self.nf), interval=interval)
        self.fig.canvas.draw_idle()
        if save:
            self.sn_rasterani = self.sn_raster[:-3]+'mp4'
            print(f"Save raster animation: {self.sn_rasterani}")
            self.ani.save(self.sn_rasterani)
            # self.updateJSON()

    def Iinversion(self, ii=None, save=True, interval=100, ani=True):
        bgcolor = "#212529"
        bg_second = "#484c4f"
        fontcolor = "#adb5bd"
        titlecolor = "#ffda6a"

        if ii is None:
            idx = self.nf//2
        else:
            idx = ii

        self.idx = idx
        A = fits.open(self.lia[idx])[0]
        B = fits.open(self.lib[idx])[0]
        time = fname2isot(self.lia[idx])

        self.nyI, self.nxI = A.data.shape[1:]
        nyB, nxB = B.data.shape[1:]
        self.figy = 8
        cx = 0.25*3.5
        fs = [(self.figy/2.78*(self.nxI/self.nyI)+cx)*4, self.figy]
        CX = cx/fs[0]
        nwv = 4
        w = 1/nwv-CX
        self.figI, self.axI = plt.subplots(4, 4, figsize=fs)
        self.figI.set_facecolor(bgcolor)

        self.tax = self.figI.add_subplot(121)
        self.tax.set_position([0,2.72/2.81,1,0.1/2.81])
        self.tax.set_facecolor(bgcolor)
        self.tax.set_axis_off()

        self.axD = self.figI.add_subplot(121)
        self.axD.set_position([0.06,0.03,0.92,0.3/2.81])
        self.axD.set_facecolor(bgcolor)
        if self.tdurI is None:
            self.tdurI = self.Iduration(self.axD)
        self.sn_inversion = join(self.saveD, time[:10].replace('-', '')+'_'+time[11:].replace(':',"")+"_inversion.png")

        yl = self.axD.get_ylim()
        self.axD.set_ylim(yl)
        self.pt = self.axD.plot([self.tdurI[idx], self.tdurI[idx]], yl, color='r')[0]

        
        self.title = self.tax.text(0.5,0.5, time, transform=self.tax.transAxes, ha='center', va='center', weight='bold', size=15, c=titlecolor)
        self.imIA = [None]*nwv
        self.imIB = [None]*nwv

        name = ['log S$_p, $', 'log S$_0, $', '$v_0, $', 'log $\\omega_0, $']
        self.Iid = [4, 13, 9, 11]
        lcmA = [cm.ha, cm.ha, plt.cm.RdBu_r, plt.cm.PuOr_r]
        lcmB = [cm.ca, cm.ca, plt.cm.RdBu_r, plt.cm.PuOr_r]

        self.caxA = [None]*nwv
        self.caxB = [None]*nwv

        for i in range(nwv):
            self.caxA[i] = self.figI.add_subplot(122)
            self.caxB[i] = self.figI.add_subplot(122)
            self.caxA[i].set_position([w+i/nwv+3e-3,1.70/2.81,CX/3.5,1.02/2.81])
            self.caxB[i].set_position([w+i/nwv+3e-3,0.58/2.81,CX/3.5,1.02/2.81*nyB/self.nyI])
            self.axI[0, i].set_position([i/nwv,0.48/2.81,w,0.1/2.81])
            self.axI[1, i].set_position([i/nwv,0.58/2.81,w,1.02/2.81*nyB/self.nyI])
            self.axI[2, i].set_position([i/nwv,1.60/2.81,w,0.1/2.81])
            self.axI[3, i].set_position([i/nwv,1.70/2.81,w,1.02/2.81])

            self.axI[0, i].text(0.5, 0.5, name[i]+r'$_{Ca II}$', transform=self.axI[0, i].transAxes, ha='center', va='center', size=12, c=fontcolor)
            self.axI[2, i].text(0.5, 0.5, name[i]+r'$_{H\alpha}$', transform=self.axI[2, i].transAxes, ha='center', va='center', size=12, c=fontcolor)

            da = A.data[self.Iid[i]]*A.header[f'scale{self.Iid[i]:02}']
            db = B.data[self.Iid[i]]*B.header[f'scale{self.Iid[i]:02}']
            wh = (da>-8)*(da < 8)
            mA = np.median(da[wh])
            stdA = da[wh].std()

            wh = (db>-8)*(db < 8)
            mB = np.median(db[wh])
            stdB = db[wh].std()
            if i == 2:
                da -= mA
                db -= mB
                self.mA = mA
                self.mB = mB

            self.imIA[i] = self.axI[3,i].imshow(da, lcmA[i], origin='lower')
            self.imIB[i] = self.axI[1,i].imshow(db, lcmB[i], origin='lower')
            if i == 2:
                self.imIA[i].set_clim([-7, 7])
                self.imIB[i].set_clim([-7, 7])
            elif i == 3:
                self.imIA[i].set_clim([mA-3*stdA, mA+3*stdA])
                self.imIB[i].set_clim([mB-3*stdB, mB+3*stdB])

            self.cbarA = self.figI.colorbar(self.imIA[i], cax=self.caxA[i])
            self.cbarA.locator = ticker.MaxNLocator(nbins=5)
            self.cbarA.ax.tick_params(colors=fontcolor)
            self.cbarA.update_ticks()
            self.cbarB = self.figI.colorbar(self.imIB[i], cax=self.caxB[i], orientation='vertical')
            self.cbarB.locator = ticker.MaxNLocator(nbins=5)
            self.cbarB.ax.tick_params(colors=fontcolor)
            self.cbarB.update_ticks()
            for j in range(4):
                self.axI[j,i].set_axis_off()
                self.axI[j,i].set_facecolor(bgcolor)

        self.figI.show()
        if save:
            print(f"Save inversion figure: {self.sn_inversion}")
            self.figI.savefig(self.sn_inversion)

        if ani:
            self.animationI(interval, save=save)
        return self.figI

    def chDataI(self, i):
        A = fits.open(self.lia[i])[0]
        B = fits.open(self.lib[i])[0]
        time = fname2isot(self.lia[i])
        nx = A.data.shape[-1]
        self.title.set_text(time)
        self.pt.set_xdata([self.tdurI[i], self.tdurI[i]])
        if self.nxI != nx:
            self.nxI = nx
            self.figI.set_figwidth(self.figy/2.78*self.nxI/self.nyI*4)
            for i in range(4):
                self.imIA[i].set_extent([-0.5, self.nxI-0.5,-0.5, self.nyI-0.5])
                self.imIB[i].set_extent([-0.5, self.nxI-0.5,-0.5, self.nyI-10-0.5])


        for i in range(4):
            da = A.data[self.Iid[i]]*A.header[f'scale{self.Iid[i]:02}']
            db = B.data[self.Iid[i]]*B.header[f'scale{self.Iid[i]:02}']
            if i == 2:
                da -= self.mA
                db -= self.mB
            self.imIA[i].set_data(da)
            self.imIB[i].set_data(db)

        self.figI.canvas.draw_idle()

    def animationI(self, interval=100, save=True):
        self.aniI = FuncAnimation(self.figI, self.chDataI, frames=np.arange(self.nf), interval=interval)
        self.figI.canvas.draw_idle()
        if save:
            self.sn_inversionani = self.sn_inversion[:-3]+'mp4'
            print(f"Save inversion animation: {self.sn_inversionani}")
            self.aniI.save(self.sn_inversionani)
            # self.updateJSON()


    # def convert2json(self):
    #     if type(arg) == list:
    #         con = """["""
    #         for ag in arg:
    #             con += f""""{ag}", """ 
    #         con = con[:-2]+"""]"""

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


def demo():
    cdir = '/Users/jhkang/Data/FISS/200730/part2/comp'
    sdir = '/Users/jhkang/Data/FISS/200730/part2/save'
    idir = '/Users/jhkang/Data/FISS/200730/part2/inversion/flip'
    events = ['transverse MHD waves', 'fibrils']
    target = "Quiet Sun - Part2"
    pos = ["20","-75"]
    pub = ["https://ui.adsabs.harvard.edu/abs/2021JKAS...54..139C", "https://ui.adsabs.harvard.edu/abs/2023ApJ...958..131K"]
    cobs = ["https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_IRIS_20200730_155928_3600011659_2020-07-30T15%3A59%3A282020-07-30T15%3A59%3A28.xml"]
    k = makeRecDataJSON(cdir, idir, sdir, events=events, position=pos, publication=pub, target=target, coobs=cobs)
    k.saveAll(excinv=True)