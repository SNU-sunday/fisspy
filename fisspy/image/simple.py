"""
"""
import numpy as np
import matplotlib.pyplot as plt
from fisspy.io import read
import fisspy
from fisspy.analysis.doppler import simple_wvcalib
from sunpy.time import TimeRange
from glob import glob
from os.path import join, basename
from . import interactive
from matplotlib import gridspec
__author__ = "Juhyeong Kang"
__email__ = "kailia@snu.ac.kr"

class FISS_data_viewer(object):
    """
    """
    
    def __init__ (self, listA= None, listB= None, fdir= None,
                  interpolation= None, scale= 'log'):
        
        if not listA and not listB and not fdir:
            ValueError('FISS file lists or fdir must be given.')
        elif fdir:
            listA = glob(join(fdir,'*A1*.fts'))
            listB = glob(join(fdir,'*A1*.fts'))
            listA.sort()
            listB.sort()
            
        if len(listA) != len(listB):
            ValueError('The given two list A and B must have' + 
                       'the same number of elements.')
        listA.sort()
        listB.sort()
        self.listA = listA
        self.listB = listB
        self.headerAs = read.getheader(self.listA[0])
        self.headerBs = read.getheader(self.listB[0])
        self.date = self.headerAs['date'][:10]
        self.start = read.getheader(self.listA[0])['date']
        self.end = read.getheader(self.listA[-1])['date']
        self.trange = TimeRange(self.start, self.end)
        self.wvA = self.headerAs['wavelen']
        self.wvA0 = self.headerAs['crval1']
        self.maxA = [False]
        self.maxB = [None]
        self.minA = [None]
        self.minB = [None]
        self.telpos = self.headerAs['tel_xpos'] * 1e3 + self.headerAs['tel_ypos']
        self.expT = self.headerAs['exptime']
        self.expTB = self.headerBs['exptime']
        self.fnum = 0
        if not interpolation:
            self.interp = 'bicubic'
        if not scale:
            self.scale = 'log'
        else:
            self.scale = scale
        if scale != 'linear' or scale != 'log':
            ValueError('Available scale is only linear and log')
        
    def IFDV (self, smooth=False):
        from skimage.viewer.widgets.core import Slider, Button, Text
        from matplotlib.widgets import Cursor
        from .coalignment import alignoffset
        from .base import shift
        try:
            from PyQt5 import QtCore
            from PyQt5.QtWidgets import QWidget, QVBoxLayout
            from PyQt5.QtWidgets import QHBoxLayout, QDockWidget, QLineEdit
            from fisspy.vc import qtvc
        except:
            from PyQt4 import QtCore
            from PyQt4.QtGui import QWidget, QVBoxLayout
            from PyQt4.QtGui import QHBoxLayout, QDockWidget, QLineEdit
            
        self.smooth = smooth
        self.wvrA = 0
        self.wvrB = 0
        self.headerA = read.getheader(self.listA[self.fnum])
        self.headerB = read.getheader(self.listB[self.fnum])
        self.frameA =  read.frame(self.listA[self.fnum], xmax=True,
                                  smooth= self.smooth)
        self.frameB =  read.frame(self.listB[self.fnum], xmax= True,
                                  smooth= self.smooth)
        self.nx = self.frameA.shape[1]
        self.shy, self.shx = alignoffset(self.frameB[:,:-1,50],
                              self.frameA[:250,:-1,-50])
        rA = read.frame2raster(self.frameA, self.headerA, self.wvrA)
        rB = read.frame2raster(self.frameB, self.headerB, self.wvrB)
        rB1 = shift(rB, (-shy, -shx))
        self.xp = 20
        self.yp = 20
        
        if self.wvA == '6562':
            self.filter_set = 1
            self.cmA = fisspy.cm.ha
            self.cmB = fisspy.cm.ca
            
        elif self.wvA == '5889':
            self.filter_set = 2
            self.cmA = fisspy.cm.na
            self.cmB = fisspy.cm.fe
            
        fig = plt.figure(figsize=(17,9))
        ax = [None]*4
        ax[0]=fig.add_subplot(141)
        ax[1]=fig.add_subplot(142)
        ax[2]=fig.add_subplot(222)
        ax[3]=fig.add_subplot(224)
        
        rasterA = ax[0].imshow(rA, origin='lower', cmap= self.cmA)
        rasterB = ax[1].imshow(rA, origin='lower', cmap= self.cmA)
        
        self.ax = ax
        clim(self, rA, rB)
        
        ax[0].set_xlabel('X position[pixel]')
        ax[1].set_xlabel('X position[pixel]')
        ax[0].set_ylabel('Y position[pixel]')
        ax[1].set_ylabel('Y position[pixel]')
        ax[1].set_ylim(-0.5,255.5)
        ax[1].set_facecolor('black')
        rasterA.set_clim(self.minA, self.maxA)
        rasterB.set_clim(self.minB, self.maxB)
        
        self.rasterA = rasterA
        self.rasterB = rasterB
        
        title(self)
        ax[0].set_title(self.titleA)
        ax[1].set_title(self.titleB)
        wvcA = simple_wvcalib(self.headerA)
        wvcB = simple_wvcalib(self.headerB)
        self.wvcA = wvcA
        self.wvcB = wvcB
        
        p1 = ax[2].plot(wvcA, self.frameA[self.yp, self.xp], color='k')
        p2 = ax[3].plot(wvcB,
               self.frameB[self.yp+int(round(self.shy[0])),
                           self.xp+int(round(self.shx[0]))], color='k')
        self.p1 = p1
        self.p2 = p2
        
        ax[2].set_title(self.titleA)
        ax[3].set_title(self.titleB)
        ax[2].set_xlabel(r'Wavelength [$\AA$]')
        ax[3].set_xlabel(r'Wavelength [$\AA$]')
        ax[2].set_ylabel(r'$I_{\lambda}$ [count]')
        ax[3].set_ylabel(r'$I_{\lambda}$ [count]')
        ax[2].set_xlim(wvcA.min(), wvcA.max())
        ax[3].set_xlim(wvcB.min(), wvcB.max())
        ax[2].set_ylim(self.frameA[self.yp, self.xp].min()-100,
                       self.frameA[self.yp, self.xp].max()+100)
        ax[3].set_ylim(self.frameB.min()-100, self.frameB.max()+100)
        

        Acur = Cursor(ax[0], color='r', useblit= True)
        Bcur = Cursor(ax[1], color='r', useblit= True)
        Apcur = Cursor(ax[2], color='r', useblit= True)
        Bpcur = Cursor(ax[3], color='r', useblit= True)
        
        root = fig.canvas.manager.window
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        hboxf = QHBoxLayout(panel)
        hboxA = QHBoxLayout(panel)
        hboxB = QHBoxLayout(panel)
        
        fnumtxt = Text('Current frame number.')
        fnum = QLineEdit()
        fnum.setText(self.fnum)
        fnum.editingFinished.connect(chframe)
        xptxt = Text('X position')
        xp = QLineEdit()
        xp.setText(self.xp)
        yptxt = Text('Y position')
        yp = QLineEdit()
        yp.setText(self.yp)
        xp.editingFinished.connect(position)
        yp.editingFinished.connect(position)
        
        hboxf.addWidget(fnumtxt)
        hboxf.addWidget(fnum)
        hboxf.addWidget(xptxt)
        hboxf.addWidget(xp)
        hboxf.addWidget(yptxt)
        hboxf.addWidget(yp)
        
        Asldm = Slider('CamA Color Range Min', 500, 
                       12000, self.minA, value_type = 'int')
        AsldM = Slider('CamA Color Range Max', 500, 
                       12000, self.minA, value_type = 'int')
        Bsldm = Slider('CamB Color Range Min', 500, 
                       12000, self.minB, value_type = 'int')
        BsldM = Slider('CamB Color Range Max', 500, 
                       12000, self.minB, value_type = 'int')
        Asldm.slider.valueChanged.connect(climA)
        AsldM.slider.valueChanged.connect(climA)
        Bsldm.slider.valueChanged.connect(climB)
        BsldM.slider.valueChanged.connect(climB)
        hboxA.addWidget(Asldm)
        hboxA.addWidget(AsldM)
        hboxB.addWidget(Bsldm)
        hboxB.addWidget(BsldM)
        
        vbox.addLayout(hboxf)
        vbox.addLayout(hboxA)
        vbox.addLayout(hboxB)
        
        fig.tight_layout()
        self.figIFDV = fig
        def title(self):
            self.titleA = r'GST/FISS %s $\AA$ %s'%(self.headerA['wavelen'],
                                               self.headerA['date'])
            self.titleB = r'GST/FISS %s $\AA$ %s'%(self.headerB['wavelen'],
                                               self.headerB['date'])
        def position(self):
            self.xp = int(xp.text())
            self.yp = int(yp.text())
            self.p1[0].set_data(self.wvcA, self.frameA[self.yp, self.xp])
            self.p2[0].set_data(self.wvcB,
                               self.frameB[self.yp+int(round(self.shy[0])),
                                           self.xp+int(round(self.shx[0]))])
            
        def chframe(self):
            # if the wvA or tel_xpos or nx is changed, 
            # then re-align the CamA and CamB
            
            self.fnum = fnum.text()
            self.frameA = read.frame(self.listA[self.fnum],
                                     xmax=True, smooth= self.smooth)
            self.frameB = read.frame(self.listB[self.fnum],
                                     xmax=True, smooth= self.smooth)
            self.headerA = read.getheader(self.listA[self.fnum])
            self.headerB = read.getheader(self.listB[self.fnum])
            wvA = self.headerA['wavelen'][:4]
            wvB = self.headerB['wavelen'][:4]
            telpos = self.headerA['tel_xpos'] * 1e3 + self.headerA['tel_ypos']
            expT = self.headerA['exptime']
            expTB = self.headerA['exptime']
            
            if wvA != self.wvA:
                self.wvA = wvA
                self.wvB = wvB
                self.wvA0 = self.headerA['crval1']
                self.wvB0 = self.headerB['crval1']
                self.minA = [False]
                self.maxA = [None]
                self.minB = [None]
                self.maxB = [None]
            if telpos != self.telpos:
                self.telpos = telpos
                self.minA = [False]
                self.minB = [None]
                self.maxA = [None]
                self.maxB = [None]
            if expT != self.expT:
                self.expT = expT
                self.minA = [False]
                self.minB = [None]
                self.maxA = [None]
                self.maxB = [None]
            if expTB != self.expTB:
                self.expTB = expTB
                self.minA = [False]
                self.minB = [None]
                self.maxA = [None]
                self.maxB = [None]
                
                
            wvcA = simple_wvcalib(self.headerA)
            wvcB = simple_wvcalib(self.headerB)
            rA = read.frame2raster(self.frameA, self.headerA, wv= self.wvrA)
            rB = read.frame2raster(self.frameB, self.headerB, wv= self.wvrB)
            
            if not self.minA[0]:
                clim(self, rA, rB)
                
            if self.nx != self.frameA.shape[1]:
                self.nx = self.frameA.shape[1]
                self.ax[0].set_xlim(-0.5, self.nx-0.5)
                self.ax[1].set_xlim(-0.5, self.nx-0.5)
            
            title(self)
            ax[0].set_title(self.titleA)
            ax[1].set_title(self.titleB)
            self.rasterA.set_data(rA)
            self.rasterB.set_data(rB)
            self.p1[0].set_data(wvcA, self.frameA[self.yp, self.xp])
            self.p2[0].set_data(wvcB,
                   self.frameB[self.yp+int(round(self.shy[0])),
                               self.xp+int(round(self.shx[0]))])
    
            self.rasterA.set_clim(self.minA, self.maxA)
            self.rasterB.set_clim(self.minB, self.maxB)
            
            
        def clim(self, rA, rB) :
            if not self.minA[0]:
                maxA = rA.max()
                maxB = rB.max()
                wh0A = rA <= 10
                rA[wh0A] = 1e4
                minA = rA.min()
                rA[wh0A] = 0
                wh0B = rB <= 10
                rB[wh0B] = 1e4
                minB = rB.min()
                rB[wh0B] = 0
                self.minA = minA
                self.minB = minB
                self.maxA = maxA
                self.maxB = maxB
            
        def mark(self, event):
            if event.key == 'm':
                axp=event.inaxes._position.get_points()[0,0]
                ayp=event.inaxes._position.get_points()[0,1]
                
        def climA(self):
            climm = Asldm.val
            climM = AsldM.val
            self.rasterA.set_clim(climm, climM)
            self.figIFDV.canvas.draw_idle()
            
        def climB(self):
            climm = Bsldm.val
            climM = BsldM.val
            self.rasterB.set_clim(climm, climM)
            self.figIFDV.canvas.draw_idle()
            
            
        
    
    def image_set (self, fnum, wvseta= None, 
                   wvsetb= None, mag= 1.5, dpi= 100,
                   mode= 'target', smooth = False):
        """
        """
        plt.style.use('dark_background')
        ps = 20
        ha = read.getheader(self.listA[fnum])
        hb = read.getheader(self.listB[fnum])
        
        self.ps = ps
        self.mag = mag
        self.dpi = dpi
        self.smooth = smooth
        
        if wvseta:
            if len(wvseta) != len(wvsetb):
                ValueError('The number of wvseta and wvsetb must be same.')
        
        
        wvA = ha['wavelen'][:4]
        wvB = hb['wavelen'][:4]
        
        if wvA != self.wvA:
            self.wvA = wvA
            self.wvB = wvB
            self.wvA0 = ha['crval1']
            self.wvB0 = hb['crval1']
            self.minA = [False]
            self.maxA = [None]
            self.minB = [None]
            self.maxB = [None]
        
        telpos = ha['tel_xpos'] * 1e3 + ha['tel_ypos']
        
        if telpos != self.telpos:
            self.telpos = telpos
            self.minA = [False]
            self.minB = [None]
            self.maxA = [None]
            self.maxB = [None]
            
        expT = ha['exptime']
        expTB = hb['exptime']
        
        if expT != self.expT:
            self.expT = expT
            self.minA = [False]
            self.minB = [None]
            self.maxA = [None]
            self.maxB = [None]
        
        if expTB != self.expTB:
            self.expTB = expTB
            self.minA = [False]
            self.minB = [None]
            self.maxA = [None]
            self.maxB = [None]
            
        if self.wvA == '6562':
            self.filter_set = 1
            self.cmA = fisspy.cm.ha
            self.cmB = fisspy.cm.ca
            
        elif self.wvA == '5889':
            self.filter_set = 2
            self.cmA = fisspy.cm.na
            self.cmB = fisspy.cm.fe
        
        if not wvseta and self.filter_set == 1:
            wvseta = [-4., -1., -0.5, 0., 0.5, 1.]
            wvsetb = [-5., -0.8, -0.2, 0., 0.2, 0.8]
            
        elif not wvseta and self.filter_set == 2:
            wvseta = [-3., -0.6, -0.1, 0., 0.1, 2.9]
            wvsetb = [-3, -1.56, -0.1, 0., 0.1, 1.3]
        
        self.wvsetA = np.array(wvseta)
        self.wvsetB = np.array(wvsetb)
        nwvset = len(wvseta)
        self.nwvset = nwvset
        rasterA = read.raster(self.listA[fnum], self.wvsetA,
                              smooth= self.smooth)
        rasterB = read.raster(self.listB[fnum], self.wvsetB,
                              smooth= self.smooth)
        if self.scale == 'log':
            rasterA = np.log(rasterA)
            rasterB = np.log(rasterB)
        if not self.maxA[0] :
            maxA = rasterA.max(axis=(1,2))
            maxB = rasterB.max(axis=(1,2))
            if self.scale == 'linear':
                wh0A = rasterA <= 10
                rasterA[wh0A] = 1e4
                minA = rasterA.min(axis=(1,2))
                rasterA[wh0A] = 0
                wh0B = rasterB <= 10
                rasterB[wh0B] = 1e4
                minB = rasterB.min(axis=(1,2))
                rasterB[wh0B] = 0
            elif self.scale == 'log':
                wh0A = rasterA <= 3
                rasterA[wh0A] = 1e4
                minA = rasterA.min(axis=(1,2))
                rasterA[wh0A] = 0
                wh0B = rasterB <= 3
                rasterB[wh0B] = 1e4
                minB = rasterB.min(axis=(1,2))
                rasterB[wh0B] = 0
            self.minA = minA
            self.minB = minB
            self.maxA = maxA
            self.maxB = maxB
        else:
            minA = self.minA
            minB = self.minB
            maxA = self.maxA
            maxB = self.maxB
            
        fs = [ha['naxis3'] * nwvset * mag / dpi, 
              (ha['naxis2'] + hb['naxis2'] + ps*5) * mag / dpi] # 20 pixels are set for time bar and etc.
        if fs[1]*dpi % 2:
            fs[1] += 1/dpi
        self.fs = fs
        # axes setting for image
        fig = plt.figure(figsize= fs, dpi= dpi)
        gs = gridspec.GridSpec(7, nwvset)
        gs.update(left= 0, right= 1, bottom= 0, top= 1,
                  wspace= 0, hspace= 0)
        
        pb1 = 1 - ps * mag /dpi / fs[1]
        pb2 = pb1 - ha['naxis2'] * mag / dpi / fs[1]
        pb3 = pb2 - ps * mag /dpi / fs[1]
        pb4 = pb3 - hb['naxis2'] * mag / dpi / fs[1]
        pb5 = pb4 - ps * mag /dpi / fs[1]
        pb6 = pb5 - ps * mag /dpi / fs[1]
        ph = ps * mag / dpi / fs[1]
        whA = ha['naxis3'] * mag / dpi /fs[0]
        whB = hb['naxis3'] * mag / dpi /fs[0]
        Ah = ha['naxis2'] * mag / dpi /fs[1]
        Bh = hb['naxis2'] * mag / dpi /fs[1]
        axt = fig.add_subplot(gs[0, :])
        axtb = fig.add_subplot(gs[5, :])
        axtt = fig.add_subplot(gs[6, :])
        axt.set_position([0, pb1, 1, ph])
        axtb.set_position([0, pb6, 1, ph])
        axtt.set_position([0, 0, 1, ph])
        axt.set_axis_off()
        axtb.set_axis_off()
        axtt.set_axis_off()
        axA = [None] * nwvset
        axB = [None] * nwvset
        axlA = [None] * nwvset
        axlB = [None] * nwvset
        imA = [None] * nwvset
        imB = [None] * nwvset
        for i in range(nwvset):
            axA[i] = fig.add_subplot(gs[1, i])
            axlA[i] = fig.add_subplot(gs[2, i])
            axB[i] = fig.add_subplot(gs[3, i])
            axlB[i] = fig.add_subplot(gs[4, i])
            axA[i].set_position([i / 6, pb2, whA, Ah])
            axB[i].set_position([i / 6, pb4, whB, Bh])
            axlA[i].set_position([i / 6, pb3, whA, ph])
            axlB[i].set_position([i / 6, pb5, whB, ph])
            axlA[i].text(0.39, 0.25, r'%.1f $\AA$'%wvseta[i],
                fontsize= 12, weight= 'bold')
            axlB[i].text(0.39, 0.25, r'%.1f $\AA$'%wvsetb[i],
                fontsize= 12, weight= 'bold')
            imA[i] = axA[i].imshow(rasterA[i],
               origin= 'lower', cmap = self.cmA, interpolation= self.interp)
            imA[i].set_clim(minA[i], maxA[i])
            imB[i] = axB[i].imshow(rasterB[i],
               origin= 'lower', cmap = self.cmB, interpolation= self.interp)
            imB[i].set_clim(minB[i], maxB[i])
            axA[i].set_axis_off()
            axB[i].set_axis_off()
            axlA[i].set_axis_off()
            axlB[i].set_axis_off()
        
        try:
            wv0pa = np.where(np.array(wvseta) == 0)[0][0]
            wv0pb = np.where(np.array(wvseta) == 0)[0][0]
            axlA[wv0pa].cla()
            axlA[wv0pa].text(0.3, 0.25, r'%.1f $\AA$'%self.wvA0,
                fontsize= 12, weight= 'bold')
            axlB[wv0pb].cla()
            axlB[wv0pb].text(0.3, 0.25, r'%.1f $\AA$'%self.wvB0,
                fontsize= 12, weight= 'bold')
            axlA[wv0pa].set_axis_off()
            axlB[wv0pb].set_axis_off()
        except:
            pass
        
        axt.text(0.4, 0.25,
                 r'GST/FISS set %i %s'%(self.filter_set, self.date),
                 fontsize=12, weight='bold')
        axtt.text(0, 0.25, self.start[11:], fontsize= 12,
                  weight= 'bold')
        axtt.text(1 - 0.08 * 10.8 / fs[0], 0.25, self.end[11:], fontsize= 12,
                  weight= 'bold')
        
        
        dtfull = self.trange.dt.seconds
        dt = TimeRange(self.start, ha['date']).dt.seconds
        
        be = dt / dtfull
        axtb.fill([0, 0, be, be], [0, 1, 1, 0], 'silver')
        axtb.set_xlim(0, 1)
        axtb.set_ylim(0, 1)
        
        if be >= 0.077 * 10.8 /fs[0] and be <= 1 - 0.153 * 10.8 /fs[0]:
            ctimetxt = axtt.text(be, 0.25, ha['date'][11:], fontsize=12,
                      weight= 'bold')
        else:
            ctimetxt = axtt.text(be, 0.25, '', fontsize=12,
                      weight= 'bold')
        
        self.fig = fig
        self.imA = imA
        self.imB = imB
        self.axtb = axtb
        self.timetext = ctimetxt
        
        if mode == 'full' and fnum < 3 :
            self.minA = [False]
            self.minB = [None]
            self.maxA = [None]
            self.maxB = [None]
            
        
    def change_img_frame (self, fnum):
        """
        """
        
        rasterA = read.raster(self.listA[fnum], self.wvsetA,
                              self.smooth)
        rasterB = read.raster(self.listB[fnum], self.wvsetB,
                              self.smooth)
        ha = read.getheader(self.listA[fnum])
        hb = read.getheader(self.listB[fnum])
        
        if self.scale == 'log':
            rasterA = np.log(rasterA)
            rasterB = np.log(rasterB)
        
        expT = ha['exptime']
        expTB = hb['exptime']
        
        if expT != self.expT:
            self.expT = expT
            self.minA = [False]
            self.minB = [None]
            self.maxA = [None]
            self.maxB = [None]
        
        if expTB != self.expTB:
            self.expTB = expTB
            self.minA = [False]
            self.minB = [None]
            self.maxA = [None]
            self.maxB = [None]
        
        for i in range(self.nwvset):
            self.imA[i].set_data(rasterA[i])
            self.imB[i].set_data(rasterB[i])
            
        if not self.maxA[0] :
            maxA = rasterA.max(axis=(1,2))
            maxB = rasterB.max(axis=(1,2))
            if self.scale == 'linear':
                wh0A = rasterA <= 10
                rasterA[wh0A] = 1e4
                minA = rasterA.min(axis=(1,2))
                rasterA[wh0A] = 0
                wh0B = rasterB <= 10
                rasterB[wh0B] = 1e4
                minB = rasterB.min(axis=(1,2))
                rasterB[wh0B] = 0
            elif self.scale == 'log':
                wh0A = rasterA <= 3
                rasterA[wh0A] = 1e4
                minA = rasterA.min(axis=(1,2))
                rasterA[wh0A] = 0
                wh0B = rasterB <= 3
                rasterB[wh0B] = 1e4
                minB = rasterB.min(axis=(1,2))
                rasterB[wh0B] = 0
            self.minA = minA
            self.minB = minB
            self.maxA = maxA
            self.maxB = maxB
            
            for i in range(self.nwvset):
                self.imA[i].set_clim(minA[i], maxA[i])
                self.imB[i].set_clim(minB[i], maxB[i])
        
        
        cdate = ha['date']
        dtfull = self.trange.dt.seconds
        dt = TimeRange(self.start, cdate).dt.seconds
        be = dt / dtfull
        
        self.axtb.cla()
        self.axtb.fill([0, 0, be, be], [0, 1, 1, 0], 'silver')
        self.axtb.set_xlim(0, 1)
        self.axtb.set_ylim(0, 1)
        
        if be >= 0.077 * 10.8 / self.fs[0] and be <= 1 - 0.153 * 10.8 / self.fs[0]:
            self.timetext.set_position([be, 0.25])
            self.timetext.set_text(cdate[11:])
        else:
            self.timetext.set_text('')
            
        self.axtb.set_axis_off()
        
        
    def savefig (self, fname, **kwargs):
        """
        """
        return self.fig.savefig(fname, dpi= self.dpi, **kwargs)
    
    def saveallfig2mkvideo (self, dirn, **kwargs):
        """
        """
        self.change_img_frame(0)
        
        for n, i in enumerate(self.listA):
            if n % 50 == 0:
                print('%i th frame'%n)
            fn = basename(i)[5:20] + '.png'
            self.change_img_frame(n)
            self.fig.savefig(join(dirn, fn), dpi= self.dpi, 
                             **kwargs)
    def saveallfig (self, dirn, **kwargs):
        
        for n, i in enumerate(self.listA):
            if n % 50 == 0:
                print('%i th frame'%n)
            fn = basename(i)[5:20] + '.png'
            self.image_set(n, mode= 'full')
            self.savefig(join(dirn, fn))
            plt.close(self.fig)
        
    def mkvideo (self, dirn, fpsi, oname):
        img = glob(join(dirn, '*.png'))        
        img.sort()
        fisspy.ffmpeg(img, fpsi, oname)
        
        