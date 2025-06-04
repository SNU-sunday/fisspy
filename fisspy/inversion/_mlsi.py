from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from multiprocessing import cpu_count
from ..read import FISS
from ._mlsi_base import Model, RadLoss, parInform, par0_3layers, ParControl, parInform, cal_3layers, cal_residue, CloudModel, get_Cloud
from ..correction import corAll, get_pure, get_centerWV, get_sel, get_Inoise, normalizeProfile
from time import time
from os.path import join, dirname, basename, isdir, isfile
from os import mkdir
from astropy.io import fits
from .. import __version__ as fv
from PyQt5 import QtGui, QtCore, QtWidgets
from ..align import alignOffset, shiftImage


def MLSI4file(ifile, ofile=None, subsec=None, ncore=-1, quiet=True):
    """
    MLSI for a FISS file.

    Parameters
    ----------
    ifile: `str`
        Input FISS file name.
    ofile: `str`, optional
        Output file name.
        The default name is the dirname(ifile)+'inv'+'*_par.fts'.
    subsec: `list`, optional
        Subsection of the image to be used for the inversion.
        [left, right, bottom, top]
    ncore: `int`, optional
        The number of CPU cores to be used for the inversion.
        The default is -1, which means using all available cores.
    quiet: `bool`, optional
        Run the function in quiet mode.
        The default is True.

    Returns
    -------
    dt: `float`
        The runtime of the inversion in seconds.

    Examples
    --------
    >>> t = MLSI4file('FISS_A1_c.fts')
    """
    nc = cpu_count()

    ncc = np.minimum(nc, ncore)
    if ncc == -1:
        ncc = nc
    ts = time()
    a = FISS(ifile)
    pa2 = corAll(a, subsec, ncore=ncc)
    if subsec is not None:
        x1, x2, y1, y2 = subsec
    else:
        x1 = 0
        x2 = a.nx
        y1 = 0
        y2 = a.ny
    if not quiet:
        t1 = time()
        dummy = Model(a.Rwave, pa2[0,0], ncore=1)
        t2 = time()

        expT = (t2-t1)*((x2-x1)*(y2-y1))/ncc*1.5
        mm = expT // 60
        print(f"It will take about {mm:.0f} + {mm*0.5:.1f} min.")
    
    p, i0, i1, i2, epsD, epsP = Model(a.Rwave, pa2, line=a.line, ncore=ncc)
    RL1, RL2 = RadLoss(p, line=a.line)

    sh = p.shape
    npar = [sh[0]+4]+list(sh[1:])
    pars = np.zeros(npar)
    pars[:15] = p
    pars[15] = epsD
    pars[16] = epsP
    pars[17] = RL2
    pars[18] = RL1

    sdir = dirname(ifile)
    sdir = join(sdir, 'inv')
    if not isdir(sdir):
        mkdir(sdir)
    if ofile is None:
        bn = basename(ifile)
        of = bn.replace('c.fts','par.fts')
        sname = join(sdir, of)
    else:
        if not dirname(ofile):
            sname = join(sdir, ofile)
        sname = ofile
    print(sname)

    pscale = np.zeros(npar)
    pscale = abs(pars).max((1,2))/32000.
    pars = (pars/pscale[:,None,None]).astype(np.int16)

    hdu = fits.PrimaryHDU(pars)
    h = hdu.header
    PI = parInform()
    h['xstart'] = (x1, 'pixel value of x=0 in orignal comp data.')
    h['ystart'] = (y1, 'pixel value of y=0 in orignal comp data.')


    for i, ps in enumerate(pscale):
        h[f'scale{i:02d}'] = (ps, f'scale factor for {PI[i][0]}')

    h['fileorig'] = basename(ifile)
    h['version'] = f"fisspy v{fv}"
    te = time()
    dt = te-ts
    h['runtime'] = (f"{dt:.0f}", "in the unit of second")
    hdu.writeto(sname, overwrite=True)


    if not quiet:
        mm = dt // 60
        ss = dt % 60
        print(f"MLSI4file-Runtime: {mm:.0f} min {ss:.0f} sec")

    return dt

class MLSI4profile:
    def __init__(self, wv, prof, rprof, line='ha', hybrid=True, title=None, sdir=None, **kwargs):
        """
        Interactive Multi Layer Spectral Inversion for a profile.

        Parameters
        ----------
        wv: `~numpy.ndarray`
            Wavelength array in A from line center.
        prof: `~numpy.ndarray`
            Profile to be fitted.
        rprof: `~numpy.ndarray`
            Reference profile.
        line: `str`, optional
            Line designation. The default is 'ha'.
        hybrid: `bool`, optional
            If True, the hybrid model (3 layers + cloud) is used. The default is True.
        title: `str`, optional
            Title of the plot. The default is None.
        par: `~numpy.ndarray`, optional
            Initial parameters for the fitting. The default is None.
        radloss: `~numpy.ndarray`, optional
            Radiative loss. The default is None.
        eps: `~numpy.ndarray`, optional
            epsD and epsP. The default is None.
        title: `str`, optional
            Title of the plot. The default is None.
        """
        import matplotlib as mpl
        mpl.rcParams['backend']='Qt5Agg'
        self.wv1 = wv
        self.prof = prof
        self.rprof = rprof
        self.line = line
        self.hybrid = hybrid
        self.par0, self.psig = par0_3layers(wv, prof, line=line)
        self.free, self.cons = ParControl(line=line)
        self.cons0 = self.cons.copy()
        self.cwv = get_centerWV(line=line)
        self.pure = get_pure(wv+self.cwv, line=line)
        self.pc = np.zeros(4)
        self.wvb = np.zeros(2)
        self.sel = self.selwv()
        self.selc = (wv-self.wvb[0])*(wv-self.wvb[1]) <= 0.
        self.sigma = get_Inoise(prof, line=self.line)
        self.sdir = sdir
        if self.sdir is None:
            self.sdir = self.ldir = QtCore.QDir.homePath()

        par = kwargs.pop('par', None)
        if par is None:
            par = self.par0.copy()
        rloss = kwargs.pop('radloss', np.zeros(2))
        eps = kwargs.pop('eps', np.zeros(2))
        self.rloss = rloss
        self.eps = eps
        self.par = par
        self.npar = len(par)
        self.modelc = prof
        self._init_plot(title=title)
        self._Clicked_Fit()

    def _init_plot(self, title):
        """
        Initialize the plot for the MLSI profile fitting.
        """
        numname = f'MLSI {self.line} profile fitting'
        self.fig = plt.figure(numname)
        self.fig.clf()
        
        mw = self.fig.canvas.manager.window
        for dock in mw.findChildren(QtWidgets.QDockWidget):
            dock.close()

        self.ax = self.fig.subplots(2, sharex=True)
        ax0, ax1 = self.ax
        
        self.iplot0 = ax0.plot(self.wv1, self.prof, 'k', linewidth=3, label=r'$I_{obs}$')[0]
        self.iplot02 = ax0.plot(self.wv1, self.prof, 'g', linewidth=1, label=r'$I_2$')[0]
        self.iplot01 = ax0.plot(self.wv1, self.prof, 'b', linewidth=1, label=r'$I_{1}$')[0]
        self.iplot00 = ax0.plot(self.wv1, self.prof, 'r',  linewidth=1, label=r'$I_{0}$')[0]

        self.iplotref = ax0.plot(self.wv1, self.rprof, 'c--', linewidth=1, label=r'$I_{ref}$')[0]
        if self.hybrid:
            self.iplot0c=ax0.plot(self.wv1, self.modelc, 'm', linewidth=1, label=r'$I_m$')[0]
        
        ax0.set_ylabel('Intensity')
        ax0.set_yscale('log')
        ax0.set_ylim([0.05, 2.])
        ax0.set_xlim([-4, 4])
        ax0.set_title(title, fontsize=12)

        self.txtsrc = ax0.text(0.05, 0.15,     
                 rf'$\log\, S_p$={self.par[4]:.2f}, $\log\, S_2$={self.par[5]:.2f}, ' + 
                 rf'$\log\, S_1$={self.par[12]:.2f}, $\log\, S_0$={self.par[13]:.2f}',
                     transform=ax0.transAxes, size=11)
        self.txtwv = ax0.text(0.05, 0.05,     
                 rf'$v_1$={self.par[8]:.2f}, $v_0$={self.par[9]:.2f}, ' + 
                 rf'$\log\, w_1$={self.par[10]:.2f}, $\log\, w_0$={self.par[11]:.2f}',
                     transform=ax0.transAxes, size=11)
        self.txteps = ax1.text(0.05, 0.05,
                  f'$\\epsilon_D$={self.eps[0]:.2f}, $\\epsilon_P$={self.eps[1]:.2f}, Radloss2={self.rloss[0]:.1f}, Radloss1={self.rloss[1]:.1f}', transform=ax1.transAxes, size=11)
        
        if self.hybrid: 
            self.txtcl = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, size=11)

        ax1.plot([-700,700], [0,0], linewidth=1, color='k')
        self.iplot1 = ax1.plot(self.wv1[self.sel], self.prof[self.sel]*0,'r.', ms=3)[0]
        if self.hybrid:
            self.iplot1c = ax1.plot(self.wv1[self.selc], self.prof[self.selc]*0, 'm.', ms=3)[0]
        ax1.set_xlabel(f'$\\lambda$ - {self.cwv:.2f} [$\\AA$]')
        ax1.set_ylabel(r'$(I_{obs}-I_0)/\sigma$')
        ax1.set_yscale('linear')
        ax1.set_ylim([-10,10])
        ax1.set_xlim([-4, 4])
        ax0.legend(loc='lower right')

        self.DockWidget()
        self._setAllitem()
        self.fig.tight_layout()
        self.fig.show()
        self.fig.set_figheight(9)
        self.fig.set_figwidth(9)
        self.fig.tight_layout()
        # self.Writepar()
        # self.Redraw()

    def selwv(self):
        """
        To re-select the wavelength to be used for the fitting by excluding the wavelengths bounded by two boundary wavelengths

        Returns
        -------
        sel: `~numpy.ndarray`
            array of selected or not.
        """
        sel = get_sel(self.wv1, self.line)*((self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) >= 0)
        return sel
    
    def dataReload(self, prof, title):
        """
        Update and reload the line profile.

        Parameters
        ----------
        prof: `~numpy.ndarray`
            New profile to be fitted.
        title: `str`
            Title of the plot.
        """
        self.prof = prof
        self.ax[0].set_title(title, fontsize=12)
        self._Clicked_Init()
        self._Clicked_Fit()

    def setFont(self):
        """
        To set the font (class) used in the widgets

        Returns
        -------
        font : class
            font class.

        """
        # 
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        return font    

    def DockWidget(self):
        """
        Configure the widgets
        """
        lpar = parInform()

        lp = [x[0] for x in lpar]
        lp[-1] = 'wvb2'
        lp[-2] = 'wvb1'
        font = self.setFont()
        self.root = self.fig.canvas.manager.window

        self.dock = dock = QtWidgets.QDockWidget("Parameters", self.root)
        self.root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock) 
        self.panel = panel = QtWidgets.QWidget()
        dock.setWidget(panel)
        vbox = QtWidgets.QVBoxLayout(panel)

        
        self.nrow = nrow = len(lp)
        self.ncol = ncol = 3
        self.parTable = QtWidgets.QTableWidget(nrow, ncol)
        self.parTable.setFont(font)
        self.parTable.setHorizontalHeaderLabels(['Par', 'Par0', 'Psig'])
        self.parTable.setVerticalHeaderLabels(lp)
        delegate = _DoubleDelegate()
        self.parTable.setItemDelegate(delegate)

        self.fP, self.cP = ParControl(line=self.line)


        for col in range(ncol):
            self.parTable.setColumnWidth(col, 62)
            for row in range(nrow):
                item = QtWidgets.QTableWidgetItem("")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                item.setBackground(QtGui.QColor(255, 255, 255))
                item.setForeground(QtGui.QColor(0, 0, 0))
                self.parTable.setItem(row, col, item)
                if row in self.fP:
                    item.setBackground(QtGui.QColor(255, 245, 219))
                    item.setForeground(QtGui.QColor(0, 0, 0))
                if row in self.cP or row >= self.npar+2:
                    item.setBackground(QtGui.QColor(222, 254, 255))
                    item.setForeground(QtGui.QColor(0, 0, 0))
                if row >= self.npar and row < self.npar+2 or (row >=self.npar+2 and col >= 1):
                    item.setBackground(QtGui.QColor(210, 210, 210))
                    item.setForeground(QtGui.QColor(80, 80, 80))
                    item.setFlags(QtCore.Qt.NoItemFlags)

        vbox.addWidget(self.parTable)
        self.GB_log = QtWidgets.QGroupBox()
        self.GB_log.setMaximumSize(QtCore.QSize(4096, 140))
        font2 = QtGui.QFont()
        font2.setFamily("Arial")
        font2.setPointSize(14)
        font2.setBold(True)
        font2.setWeight(50)
        self.GB_log.setFont(font2)
        self.GB_log.setTitle('Log')
        hl_log2 = QtWidgets.QHBoxLayout(self.GB_log)
        scrollArea = QtWidgets.QScrollArea(self.GB_log)
        scrollArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        scrollArea.setWidgetResizable(True)
        sa_contents = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(sa_contents)
        self.L_log = QtWidgets.QLabel(self.GB_log)
        self.L_log.setFont(font)
        self.L_log.setWordWrap(True)
        self.L_log.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.L_log.setText('MLSI for the FISS line profile.')
        hl.addWidget(self.L_log)
        scrollArea.setWidget(sa_contents)
        hl_log2.addWidget(scrollArea)
        vbox.addWidget(self.GB_log)
        hbox = QtWidgets.QHBoxLayout()
        self.B_Apply = QtWidgets.QPushButton('Apply')
        self.B_Apply.setFont(font)
        self.B_Apply.clicked.connect(self._Clicked_Apply)
        hbox.addWidget(self.B_Apply)
        self.B_Init = QtWidgets.QPushButton('Initialization')
        self.B_Init.setFont(font)
        self.B_Init.clicked.connect(self._Clicked_Init)
        hbox.addWidget(self.B_Init)
        self.B_Fit = QtWidgets.QPushButton('Fit')
        self.B_Fit.setFont(font)
        self.B_Fit.clicked.connect(self._Clicked_Fit)
        hbox.addWidget(self.B_Fit)
        hbox2 = QtWidgets.QHBoxLayout()
        self.B_Save = QtWidgets.QPushButton('Save')
        self.B_Save.setFont(font)
        self.B_Save.clicked.connect(self._Clicked_Save)
        hbox2.addWidget(self.B_Save)
        self.B_Load = QtWidgets.QPushButton('Load')
        self.B_Load.setFont(font)
        self.B_Load.clicked.connect(self._Clicked_Load)
        hbox2.addWidget(self.B_Load)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)


    def _getWVB(self):
        self.wvb[0] = self._getParValue(self.npar+2, 0)
        self.wvb[1] = self._getParValue(self.npar+3, 0)
        self.sel = self.selwv()
        self.selc = (self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) <= 0.
        if self.hybrid and (self.wvb[0] != self.wvb[1]):
            self.cons = np.append(self.cons0, 13)
        else:
            self.cons = self.cons0.copy()

    def _getParValue(self, row, col):
        return float(self.parTable.item(row, col).text())
    
    def _setParValue(self, row, col, value):
        self.parTable.item(row, col).setText(f"{value:.3f}")

    def _setPar(self):
        for row in range(self.npar):
            self._setParValue(row, 0, self.par[row])

        self._setParValue(self.npar, 0, self.eps[0])
        self._setParValue(self.npar+1, 0, self.eps[1])
        self._setParValue(self.npar+2, 0, self.wvb[0])
        self._setParValue(self.npar+3, 0, self.wvb[1])

    def _setPar0(self):
        for row in range(self.npar):
            self._setParValue(row, 1, self.par0[row])
    
    def _setPsig(self):
        for row in range(self.npar):
            self._setParValue(row, 2, self.psig[row])

    def _setAllitem(self):
        self._setPar()
        self._setPar0()
        self._setPsig()

    def _Clicked_Apply(self):
        for row in range(self.npar):
            self.par[row] = self._getParValue(row, 0)
        self._getWVB()
        I0, I1, I2 = cal_3layers(self.wv1, self.par, line=self.line)
        self.iplot0.set_ydata(self.prof)                 
        self.iplot00.set_ydata(I0)
        self.iplot00.set_xdata(self.wv1)
        self.iplot02.set_ydata(I2)
        self.iplot01.set_ydata(I1)

        if (self.wvb[0] != self.wvb[1]) and self.hybrid:
            resDs = (self.prof[self.selc]-self.modelc[self.selc])/self.sigma[self.selc]
            Ndatas = len(self.wv1[self.selc]) 
            self.iplot0c.set_ydata(self.modelc[self.selc])
            self.iplot0c.set_xdata(self.wv1[self.selc])
            self.iplot1c.set_ydata(resDs)
            self.iplot1c.set_xdata(self.wv1[self.selc])
            epsDs = np.sqrt((resDs**2).mean())

        res = cal_residue(self.par[self.free], self.wv1[self.sel], self.prof[self.sel],  self.par0, self.psig, self.par, self.line, self.free, self.cons)
        Ndata = len(self.wv1[self.sel])
        epsD = np.sqrt((res[:Ndata]**2).sum())
        epsP = np.sqrt((res[Ndata:]**2).sum())
        self.eps = np.array([epsD, epsP])

        if self.wvb[0] == self.wvb[1] and self.hybrid:
            self.iplot0c.set_ydata(self.wv1[self.sel]*0)
            self.iplot0c.set_xdata(self.wv1[self.sel])
            self.iplot1c.set_ydata(self.wv1[self.sel]*0-100.)
            self.iplot1c.set_xdata(self.wv1[self.sel])

        self.iplot1.set_xdata(self.wv1[self.sel])
        self.iplot1.set_ydata(res[0:Ndata]*np.sqrt(Ndata))

        self._setParValue(self.npar, 0, self.eps[0])
        self._setParValue(self.npar+1, 0, self.eps[1])

        self.txtsrc.set_text(rf'$\log\, S_p$={self.par[4]:.2f}, $\log\, S_2$={self.par[5]:.2f}, $\log\, S_1$={self.par[12]:.2f}, $\log\, S_0$={self.par[13]:.2f}')
        self.txtwv.set_text(rf'$v_1$={self.par[8]:.2f}, $v_0$={self.par[9]:.2f}, $\log\, w_1$={self.par[10]:.2f}, $\log\, w_0$={self.par[11]:.2f}')
        self.txteps.set_text(f'$\\epsilon_D$={self.eps[0]:.2f}, $\\epsilon_P$={self.eps[1]:.2f}, Radloss2={self.rloss[0]:.1f}, Radloss1={self.rloss[1]:.1f}')

        if self.hybrid and (self.wvb[0] != self.wvb[1]):
            epsD = self.eps[0]
            epsDt = np.sqrt((epsD**2*Ndata + epsDs**2*Ndatas)/(Ndata+Ndatas))
            self.txtcl.set_text(rf'$\log\, S$={self.pc[0]:.2f}, $\log\, \tau$={self.pc[1]:.2f}, $\log\, w$={self.pc[2]:.3f}, $v$={self.pc[3]:.1f}, $\epsilon_D$={epsDs:.2f} $\epsilon_t$={epsDt:.2f}')
        else:
            self.txtcl.set_text('')

        self.fig.canvas.draw_idle()
        self.L_log.setText(f"Apply and redraw the profile with the current parameters")

    def _Clicked_Init(self):
        self.par0, self.psig = par0_3layers(self.wv1, self.prof, line=self.line)
        self.par = self.par0.copy()
        self.wvb = np.zeros(2)
        self._setAllitem()
        self._Clicked_Apply()
        self.L_log.setText(f"Initialization")


    def _Clicked_Fit(self):
        for row in range(self.npar):
            self.par[row] = self._getParValue(row, 0)
            self.par0[row] = self._getParValue(row, 1)
            self.psig[row] = self._getParValue(row, 2)
        self._getWVB()

        par, i0, i1, i2, epsD, epsP = Model(self.wv1, self.prof, sel=self.sel, par=self.par, par0=self.par0, psig=self.psig, free=self.free, constr=self.cons, line=self.line)
        RL1, RL2 = RadLoss(par, line=self.line)
        self.par = par
        self.rloss = np.array([RL2, RL1])
        self.eps = np.array([epsD, epsP])
        self._setPar()

        if self.hybrid and (self.wvb[0] != self.wvb[1]):
            self.selc = (self.wv1-self.wvb[0])*(self.wv1-self.wvb[1]) <= 0.
            pc, modelc = CloudModel(self.wv1[self.selc], i0[self.selc], self.prof[self.selc], line=self.line)
            modelc = get_Cloud(self.wv1, pc, i0, line=self.line)
            self.pc = pc
            self.modelc = modelc
        self._Clicked_Apply()
        self.L_log.setText(f"Fit the line profile")

    def _Clicked_Save(self):
        fpar = join(self.sdir, f'{self.line}_ITLM.npz')
        np.savez(fpar, par=self.par, par0=self.par0, psig=self.psig, wv=self.wv1, prof=self.prof)
        self.L_log.setText(f"Save file: \nfilename = '{fpar}'")

    def _Clicked_Load(self):
        fpar = join(self.sdir, f'{self.line}_ITLM.npz')
        if not isfile(fpar):
            self.L_log.setText(f"{fpar} is not found.")
        else:
            self.L_log.setText(f"Open file: \nfilename = '{fpar}'")
            res = np.load(fpar)
            self.par = res['par']
            self.par0 = res['par0']
            self.psig = res['psig']
            self.wv1 = res['wv']
            self.prof = res['prof']
            self._setAllitem()
            self._Clicked_Apply()

class IMLSI:
    def __init__(self, f, f2=None, x=None, y=None, sdir=None):
        """
        Interactive Multi Layer Spectral Inversion for a profile with Graphical User Interface

        Parameters
        ----------
        f: `str`
            Input FISS file name. Either Ha or Ca is available.
        f2: `str`, optional
            Input 2nd FISS file name. Either Ha or Ca is available.
        x: `float`, optional
            x position in Mm.
        y: `float`, optional
            y position in Mm.
        sdir: `str`, optional
            Direcotry name for uptput parameter table.

        Interactive key
        ---------------
        spacebar: Change the profile at the position on the mouse position.
        left, right, down, up: Shifting the position by one pixel. 
        1~6: Change the raster image at the selected wavelength based on the mouse cursor position.
        """
        if sdir is None:
            self.sdir = dirname(f)
        else:
            self.sdir = sdir
        self.fiss2 = None
        self.fiss1 = fiss1 = FISS(f)
        normalizeProfile(self.fiss1)
        self.dx = fiss1.dx * 0.725
        self.dy = fiss1.dy * 0.725
        self.nx = fiss1.nx
        self.ny = fiss1.ny
        xp = self.nx//2
        yp = self.ny//2
        xx, yy = self._pix2Mm(xp, yp, cam=0)
        self.lnum = ['1', '2', '3', '4', '5', '6']

        if x is None:
            self.x = xx
        else:
            self.x = x
        if y is None:
            self.y = yy
        else:
            self.y = y
        
        self.xp, self.yp = self._Mm2Pix(self.x, self.y, cam=0)

        if f2 is not None:
            self.fiss2 = fiss2 = FISS(f2)
            normalizeProfile(self.fiss2)
            ny = np.minimum(fiss1.ny, fiss2.ny)
            self.sh = alignOffset(fiss2.data[:ny,2:-2,50], fiss1.data[:ny,2:-2,50])[:,0]
            self.dwv2 = self.fiss2.Rwave[1] - self.fiss2.Rwave[0]
            self.xp2, self.yp2 = self._Mm2Pix(self.x, self.y, cam=1)
            self.x2, self.y2 = self._pix2Mm(self.xp2, self.yp2, cam=1)

        self.imshow()    
        

    def _pix2Mm(self, x, y, cam=0):
        if cam == 0:
            return x * self.dx, y * self.dy
        elif cam == 1:
            return (x - self.sh[1]) * self.dx, (y - self.sh[0]) * self.dy

    def _Mm2Pix(self, x, y, cam=0):
        if cam == 0:
            return int(round(x / self.dx)), int(round(y / self.dy))
        elif cam == 1:
            return int(round(x / self.dx + self.sh[1])), int(round(y / self.dy + self.sh[0]))
        
    def imshow(self):
        self._plotSetting()
        self.initRaster()

    def initRaster(self):
        self.Rextent = [-0.5*self.dx, (self.nx-0.5)*self.dx, -0.5*self.dy, (self.ny-0.5)*self.dy]
        self.dwv = self.fiss1.Rwave[1] - self.fiss1.Rwave[0]
        self.Sextent = [self.fiss1.Rwave.min()-0.5*self.dwv, self.fiss1.Rwave.max()+0.5*self.dwv, -0.5*self.dy, (self.ny-0.5)*self.dy]
        self.rwv = [-1, -0.5, 0, 0.5, 1, 4]
        nrwv = len(self.rwv)
        self.ax2 = [None]*nrwv
        fx = self.nx/50*nrwv*.65
        fy = 7
        nrow = 2
        self.xmin = self.Rextent[0]+0.5*self.dx
        self.xmax = self.Rextent[1]-0.5*self.dx
        self.ymin = self.Rextent[2]+0.5*self.dy
        self.ymax = self.Rextent[3]-0.5*self.dy
        if self.fiss2 is not None:
            self.rwv2 = [-0.5, -0.25, 0, 0.25, 0.5, 4]
            self.dwv2 = self.fiss2.Rwave[1] - self.fiss2.Rwave[0]
            self.Rextent2 = [(-0.5 - self.sh[1]) * self.dx,
                             (self.fiss2.nx - 0.5 -self.sh[1]) * self.dx,
                             (-0.5 - self.sh[0]) * self.dy,
                             (self.fiss2.ny -0.5 -self.sh[0])*self.dy]
            self.Sextent2 = [self.fiss2.Rwave.min()-0.5*self.dwv2,
                             self.fiss2.Rwave.max()+0.5*self.dwv2,
                             (-0.5 - self.sh[0]) * self.dy,
                             (self.fiss2.ny -0.5 -self.sh[0])*self.dy]
            self.xmin = min(self.xmin, self.Rextent2[0]+0.5*self.dx)
            self.xmax = max(self.xmax, self.Rextent2[1]-0.5*self.dx)
            self.ymin = min(self.ymin, self.Rextent2[2]+0.5*self.dy)
            self.ymax = max(self.ymax, self.Rextent2[3]-0.5*self.dy)
            
            self.im2 = [None]*nrwv
            fy = 10
            nrow = 3

        gs = GridSpec(nrow, 12)
        self.fig = plt.figure('Raster', figsize=[fx, fy])
        self.fig.clf()
        self.cmap = cmap = plt.get_cmap('brg')
        self.ax = [None]*nrwv
        self.im = [None]*nrwv
        self.axS = [None]*(nrow-1)
        self.imS = [None]*(nrow-1)
        self.P1 = [None]*nrwv
        self.yline1 = None
        self.wlines1 =[None]*nrwv
        
        if self.fiss2 is not None:
            self.axS[0] = self.fig.add_subplot(gs[nrow-1, :6])
        else:
            self.axS[0] = self.fig.add_subplot(gs[nrow-1, 3:9])
        self.imS[0] = self.axS[0].imshow(self.fiss1.data[:,self.xp], self.fiss1.cmap, origin='lower', interpolation='nearest', aspect='auto', extent=self.Sextent)
        self.axS[0].set_ylabel('Y (Mm)')
        self.axS[0].set_xlabel(f'$\\lambda$ - {self.fiss1.cwv:.2f} ($\\AA$)')
        if self.fiss1.line.lower() == 'ha':
            spc = r"H$\alpha$"
        elif self.fiss1.line.lower() == 'ca':
            spc = "Ca II 8542 $\\AA$"
        self.axS[0].set_title(f"Spectrogram {spc}")
        
        for i in range(nrwv):
            cc = cmap(i / (nrwv - 1))
            if i == 0:
                self.ax[i] = self.fig.add_subplot(gs[0, i*2:(i+1)*2], sharey=self.axS[0])
            else:
                self.ax[i] = self.fig.add_subplot(gs[0, i*2:(i+1)*2], sharey=self.ax[0], sharex=self.ax[0])
                self.ax[i].tick_params(labelleft=False, labelbottom=False)
            tmp = self.fiss1.getRaster(self.rwv[i]+self.fiss1.cwv)
            tt = tmp.flatten()
            tt.sort()
            ntt = len(tt)
            m = tt[int(ntt*0.001)] # 0.1 % min
            M = tt[-int(ntt*0.001)] # 0.1 % max
            self.im[i] = self.ax[i].imshow(tmp, self.fiss1.cmap, extent=self.Rextent, origin='lower', interpolation='nearest', clim=(m, M))
            self.ax[i].set_title(f'{self.rwv[i]:.2f} $\\AA$', color=cc)
            self.P1[i] = self.ax[i].plot(self.x, self.y, marker='+', color='r', mew=1.5, ms=10)[0]
            self.wlines1[i] = self.axS[0].plot((self.rwv[i],self.rwv[i]), [-100,100], color=cc, ls=(0, (5, 10)))[0]

        self.ax[0].set_xlabel('X (Mm)')
        self.ax[0].set_ylabel('Y (Mm)')
        self.yline1 = self.axS[0].plot(self.Sextent[0:2], [self.y, self.y], color='cyan', lw=1.5, ls='dashed')[0]
        prof = corAll(self.fiss1, [self.xp, self.xp+1, self.yp, self.yp+1])
        title = f"(x,y) = ({self.x:.2f}, {self.y:.2f}) Mm / ({self.xp}, {self.yp}) pix"
        self.mlsi1 = MLSI4profile(self.fiss1.Rwave, prof, self.fiss1.refProfile, line=self.fiss1.line, hybrid=True, title=title, sdir=self.sdir)


        if self.fiss2 is not None:
            self.yline2 = None
            self.wlines2 =[None]*nrwv
            self.P2 = [None]*nrwv

            self.axS[1] = self.fig.add_subplot(gs[nrow-1, 6:], sharey=self.ax[0])
            self.axS[1].tick_params(labelleft=False)
            self.imS[1] = self.axS[1].imshow(self.fiss2.data[:,self.xp2], self.fiss2.cmap, origin='lower', interpolation='nearest', aspect='auto', extent=self.Sextent2)
            if self.fiss2.line.lower() == 'ha':
                spc = r"H$\alpha$"
            elif self.fiss2.line.lower() == 'ca':
                spc = "Ca II 8542 $\\AA$"
            self.axS[1].set_title(f"Spectrogram {spc}")
            self.axS[1].set_xlabel(f'$\\lambda$ - {self.fiss2.cwv:.2f} ($\\AA$)')
            for i in range(nrwv):
                cc = cmap(i / (nrwv - 1))
                self.ax2[i] = self.fig.add_subplot(gs[1, i*2:(i+1)*2], sharey=self.ax[0], sharex=self.ax[0])
                if i != 0:
                    self.ax2[i].tick_params(labelleft=False, labelbottom=False)
                tmp = self.fiss2.getRaster(self.rwv2[i]+self.fiss2.cwv)
                tt = tmp.flatten()
                tt.sort()
                ntt = len(tt)
                m = tt[int(ntt*0.001)] # 0.1 % min
                M = tt[-int(ntt*0.001)] # 0.1 % max
                self.im2[i] = self.ax2[i].imshow(tmp, self.fiss2.cmap, extent=self.Rextent2, origin='lower', interpolation='nearest', clim=(m, M))
                self.ax2[i].set_title(f'{self.rwv2[i]:.2f} $\\AA$', color=cc)
                self.P2[i] = self.ax2[i].plot(self.x2, self.y2, marker='+', color='r', mew=1.5, ms=10)[0]
                self.wlines2[i] = self.axS[1].plot((self.rwv2[i],self.rwv2[i]), [-100,100], color=cc, ls=(0, (5, 10)))[0]

            self.ax2[0].set_xlabel('X (Mm)')
            self.ax2[0].set_ylabel('Y (Mm)')
            self.yline2 = self.axS[1].plot(self.Sextent2[0:2], [self.y2, self.y2], color='cyan', lw=1.5, ls='dashed')[0]

            prof = corAll(self.fiss2, [self.xp2, self.xp2+1, self.yp2, self.yp2+1])
            title = f"(x,y) = ({self.x2:.2f}, {self.y2:.2f}) Mm / ({self.xp2}, {self.yp2}) pix"
            self.mlsi2 = MLSI4profile(self.fiss2.Rwave, prof, self.fiss2.refProfile, line=self.fiss2.line, hybrid=True, title=title, sdir=self.sdir)

            self.axS[0].set_ylim(min(self.Rextent[2], self.Rextent2[2]), max(self.Rextent[3], self.Rextent2[3]))
        self.axS[0].set_ylim(self.Rextent[2], self.Rextent[3])
        self.fig.tight_layout(pad=0.5)
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)
        self.fig.show()

    def _onKey(self, event):
        if event.key == ' ' and (event.inaxes in self.ax or event.inaxes in self.ax2):
            x = event.xdata
            y = event.ydata
            self._updatePos(x, y)

        elif event.key == ' ' and event.inaxes in self.axS:
            y = event.ydata
            self._updatePos(self.x, y)

        elif event.key == 'left':
            self.x -= self.dx
            if self.x < self.xmin:
                self.x = self.xmax
            self._updatePos(self.x, self.y)

        elif event.key == 'right':
            self.x += self.dx
            if self.x > self.xmax:
                self.x = self.xmin
            self._updatePos(self.x, self.y)

        elif event.key == 'down':
            self.y -= self.dy
            if self.y < self.ymin:
                self.y = self.ymax
            self._updatePos(self.x, self.y)

        elif event.key == 'up':
            self.y += self.dy
            if self.y > self.ymax:
                self.y = self.ymin
            self._updatePos(self.x, self.y)

        elif event.key in self.lnum and event.inaxes in self.axS:
            idx = int(event.key) - 1
            wv = event.xdata
            if event.inaxes == self.axS[0]:
                cam = 0
            elif event.inaxes == self.axS[1]:
                cam = 1
            self._updateWV(idx, wv, cam=cam)

    def _updatePos(self, x, y):
        self.xp, self.yp = self._Mm2Pix(x, y, cam=0)
        self.x, self.y = self._pix2Mm(self.xp, self.yp, cam=0)
        
        for i in range(len(self.P1)):
            self.P1[i].set_xdata([self.x])
            self.P1[i].set_ydata([self.y])
            
        self.yline1.set_ydata([self.y, self.y])

        prof = corAll(self.fiss1, [self.xp, self.xp+1, self.yp, self.yp+1])
        title = f"(x,y) = ({self.x:.2f}, {self.y:.2f}) Mm / ({self.xp}, {self.yp}) pix"
        self.mlsi1.dataReload(prof, title)
        # self.mlsi1 = MLSI4profile(self.fiss1.Rwave, prof, self.fiss1.refProfile, line=self.fiss1.line, hybrid=True, title=title, sdir=self.sdir)
        
        if self.fiss2 is not None:
            self.xp2, self.yp2 = self._Mm2Pix(self.x, self.y, cam=1)
            self.x2, self.y2 = self._pix2Mm(self.xp2, self.yp2, cam=1)
            for i in range(len(self.rwv2)):
                self.P2[i].set_xdata([self.x2])
                self.P2[i].set_ydata([self.y2])
            self.yline2.set_ydata([self.y2, self.y2])
            prof = corAll(self.fiss2, [self.xp2, self.xp2+1, self.yp2, self.yp2+1])
            title = f"(x,y) = ({self.x2:.2f}, {self.y2:.2f}) Mm / ({self.xp2}, {self.yp2}) pix"
            # self.mlsi2 = MLSI4profile(self.fiss2.Rwave, prof, self.fiss2.refProfile, line=self.fiss2.line, hybrid=True, title=title, sdir=self.sdir)
            self.mlsi2.dataReload(prof, title)

        self.fig = plt.figure('Raster')
        self.fig.show()
        self.fig.canvas.draw_idle()

    def _updateWV(self, idx, wv, cam):
        cc = self.cmap(idx / (6 - 1))
        if cam == 0:
            self.wlines1[idx].set_xdata((wv, wv))
            self.rwv[idx] = wv
            tmp = self.fiss1.getRaster(wv+self.fiss1.cwv)
            tt = tmp.flatten()
            tt.sort()
            ntt = len(tt)
            m = tt[int(ntt*0.001)] # 0.1 % min
            M = tt[-int(ntt*0.001)] # 0.1 % max
            self.im[idx].set_data(tmp)
            self.im[idx].set_clim(m, M)
            self.ax[idx].set_title(f'{wv:.2f} $\\AA$', color=cc)
        if cam == 1:
            self.wlines2[idx].set_xdata((wv, wv))
            self.rwv2[idx] = wv
            tmp = self.fiss2.getRaster(wv+self.fiss2.cwv)
            tt = tmp.flatten()
            tt.sort()
            ntt = len(tt)
            m = tt[int(ntt*0.001)] # 0.1 % min
            M = tt[-int(ntt*0.001)] # 0.1 % max
            self.im2[idx].set_data(tmp)
            self.im2[idx].set_clim(m, M)
            self.ax2[idx].set_title(f'{wv:.2f} $\\AA$', color=cc)
        self.fig.canvas.draw_idle()

    def _plotSetting(self):
        import matplotlib as mpl
        mpl.rcParams['backend']='Qt5Agg'
        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass


class _DoubleDelegate(QtWidgets.QStyledItemDelegate):
    
    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator.setDecimals(3)
        editor.setValidator(validator)
        return editor

    def setEditorData(self, editor, index):
        value = index.model().data(index, 0) 
        editor.setText(str(value))

    def setModelData(self, editor, model, index):
        text = editor.text()
        model.setData(index, float(text))