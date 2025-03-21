from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from ..read import FISS
from ._mlsi_base import Model, RadLoss, parInform, par0_3layers, ParControl, parInform
from ..correction import corAll, get_pure, get_centerWV, get_sel, get_Inoise
from time import time
from os.path import join, dirname, basename, isdir
from os import mkdir
from astropy.io import fits
from .. import __version__ as fv
from PyQt5 import QtGui, QtCore, QtWidgets

def MLSI4file(ifile, ofile=None, subsec=None, ncore=-1, quiet=True):
    """
    """
    ts = time()
    a = FISS(ifile)
    pa2 = corAll(a, subsec)
    nc = cpu_count()

    ncc = np.minimum(nc, ncore)
    if ncc == -1:
        ncc = nc
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
    RL1, RL2 = RadLoss(p, line='ha')

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
    

class IMLSI:
    def __init__(self, f, f2=None, x=None, y=None, ):
        """
        Multi Layer Spectral Inversion for a profile with Graphical User Interface
        """
        pass

class MLSI4profile:
    def __init__(self, wv, prof, rprof, line='ha', hybrid=True, title=None, **kwargs):
        """Interactive Multi Layer Spectral Inversion"""
        import matplotlib as mpl
        mpl.rcParams['backend']='Qt5Agg'
        self.wv1 = wv
        self.prof = prof
        self.rprof = rprof
        self.line = line
        self.hybrid = hybrid
        self.par0, self.psig = par0_3layers(wv, prof, line=line)
        self.free, self.cons = ParControl(line=line)
        self.cwv = get_centerWV(line=line)
        self.pure = get_pure(wv+self.cwv, line=line)
        self.pc = np.zeros(4)
        self.wvb = np.zeros(2)
        self.sel = self.selwv()
        self.selc = (wv-self.wvb[0])*(wv-self.wvb[1]) <= 0.
        self.sigma = get_Inoise(prof, line=self.line)

        par = kwargs.pop('par', self.par0)
        rloss = kwargs.pop('radloss', np.zeros(2))
        eps = kwargs.pop('eps', np.zeros(2))
        self.rloss = rloss
        self.eps = eps
        self.par = par
        self.modelc = prof
        self._init_plot(title=title)

    def _init_plot(self, title):
        num = 1
        fl = plt.get_figlabels()
        numname = 'MLSI profile fitting'
        self.fig = plt.figure(numname, figsize=[12, 8])
        self.fig.clf()
        self.ax = self.fig.subplots(2, sharex=True)
        ax0, ax1 = self.ax
        
        self.iplot0 = ax0.plot(self.wv1, self.prof, 'k', linewidth=3, label=r'$I_{obs}$')[0]
        self.iplot02 = ax0.plot(self.wv1, self.prof, 'g', linewidth=1, label=r'$I_2$')[0]
        self.iplot01 = ax0.plot(self.wv1, self.prof, 'b', linewidth=1, label=r'$I_{1}$')[0]
        self.iplot00 = ax0.plot(self.wv1, self.prof, 'r',  linewidth=1, label=r'$I_{0}$')[0]

        if self.hybrid:
            self.iplot0c=ax0.plot(self.wv1, self.modelc, 'm', linewidth=1, label=r'$I_m$')[0]
        else:
            self.iplotref = ax0.plot(self.wv1, self.rprof, 'c--', linewidth=1, label=r'$I_{ref}$')[0]
        
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
                  r'$\epsilon_D$ ='+f'{self.eps[0]:.2f}, '+r'$\epsilon_P$'+f'={self.eps[1]:.2f}', transform=ax1.transAxes, size=11)
        
        if self.hybrid: 
            self.txtcl = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, size=11)

        ax1.plot([-700,700], [0,0], linewidth=1)
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
        self.fig.tight_layout(pad=3)
        self.fig.show()
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
        lpar = parInform()
        lp = [x[0] for x in lpar]
        font = self.setFont()
        self.root = self.fig.canvas.manager.window

        dock = QtWidgets.QDockWidget("Parameters", self.root)
        self.root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock) 
        panel = QtWidgets.QWidget()
        dock.setWidget(panel)
        vbox = QtWidgets.QVBoxLayout(panel)

        self.nrow = nrow = len(self.par)
        self.ncol = ncol = 3
        self.parTable = QtWidgets.QTableWidget(nrow, ncol)
        self.parTable.setFont(font)
        # self.parTable.setColumnCount(3)
        # self.parTable.setRowCount(len(self.par))
        self.parTable.setHorizontalHeaderLabels(['Par', 'Par0', 'Psig'])
        self.parTable.setVerticalHeaderLabels(lp)
        delegate = _DoubleDelegate()
        self.parTable.setItemDelegate(delegate)


        for col in range(ncol):
            self.parTable.setColumnWidth(col, 65)
            for row in range(nrow):
                item = QtWidgets.QTableWidgetItem(f"{col*10+row}")  # 기본값 설정
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.parTable.setItem(row, col, item)

        vbox.addWidget(self.parTable)

    def getParValue(self):
        for c in range(self.ncol):
            for r in range(self.nrow):
                item = self.parTable.item(r,c).text()


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