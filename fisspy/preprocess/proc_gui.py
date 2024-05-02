from PyQt5 import QtWidgets, QtCore, QtGui
from os.path import join, basename, isdir
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ..preprocess import proc_base
from ..analysis.wavelet import Wavelet
from glob import glob
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.time import Time
from os import makedirs
from fisspy import cm
from scipy.signal import find_peaks
from statsmodels.tsa.ar_model import AutoReg

__author__ = "Juhyung Kang"
__all__ = ["prepGUI"]

def qSleep(sec):
    ms = int(sec*1e3)
    qtTimerLoop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, qtTimerLoop.quit)
    qtTimerLoop.exec_()

class prepGUI:
    def __init__(self, basedir, ffocA=None, ffocB=None, savedir=None):
        mpl.use('Qt5Agg')
        # color
        self.bg_primary = "#212529"
        self.bg_second = "#343a40"
        self.font_normal = "#adb5bd"
        self.font_primary = "#ffda6a"
        self.font_second = "#bedcfd"
        self.font_third = "#75b798"
        self.font_err = "#ea868f"
        self.border = "#5a626a"
        self.btn_1 = "#997404"
        self.btn_2 = "#ea868f"
        self.runCam = 0 # 0: both, 1: A, 2: B

        plt.rcParams['text.color'] = self.font_normal
        plt.rcParams['axes.labelcolor'] = self.font_normal
        plt.rcParams['axes.facecolor'] = self.bg_second
        plt.rcParams['axes.edgecolor'] = self.font_normal
        plt.rcParams['xtick.color'] = self.font_normal
        plt.rcParams['ytick.color'] = self.font_normal
        plt.rcParams['axes.titlecolor'] = self.font_primary
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.facecolor'] = self.bg_primary

        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 15

        self.rcaldir = join(basedir, 'cal')
        self.pcaldir = join(basedir, 'proc', 'cal')
        if savedir is None:
            self.procdir = join(basedir, 'proc')
            self.compdir = join(basedir, 'comp')
        else:
            self.procdir = join(savedir, 'proc')
            self.compdir = join(savedir, 'comp')
        self.rawdir = join(basedir, 'raw')
        self.ffocA = ffocA
        self.ffocB = ffocB
        self.fflatL = glob(join(self.rcaldir, '*_Flat.fts'))
        self.fflatL.sort()
        self.fflatGBL = [None] * len(self.fflatL)

        self.xFringe = None
        self.yFringe = None
        self.p_s7_comp = None
        self.p_s7_prof = None
        self.stop = False

        for i,f in enumerate(self.fflatL):
            self.fflatGBL[i] = basename(f)

        self.fflatL.sort()
        self.stepNum = 0
        self.subStepNum = -1

        try:
            plt.rcParams['keymap.all_axes'].remove('a')
            plt.rcParams['keymap.save'].remove('s')
            plt.rcParams['keymap.back'].remove('c')
        except:
            pass

        self.List_step = ["Step 0: Input File",
                          "Step 1: Tilt Correction",
                          "Step 2: 1st Curvature Correction",
                          "Step 3: Fringe Subtraction",
                          "Step 4: 2nd Curvature Correction",
                          "Step 5: Make Flat",
                          "Step 6: Save Flat",
                          "Step 7: Run Preprocess"]
        
        self.List_subStep = ["3-1: Atlas Subtraction",
                             "3-2: Vertical Fringe",
                             "3-3: Spectrum Mask",
                             "3-4: Horizontal Frigne"]
        
        self.initS1 = True
        self.initS2 = True
        self.initS3_1 = True
        self.initS3_2 = True
        self.initS3_3 = True
        self.initS3_4 = True
        self.initS4 = True
        self.initS5 = True

        self.nStep = len(self.List_step)
        self.nsubStep = len(self.List_subStep)
        
        self.fig = plt.figure(figsize=[17,7])

        self.ax_pos = [[[0.06,0.07,0.88,0.86]],
                       [[0.06,0.07,0.435,0.86], [0.555,0.07,0.435,0.86]],
                       [[0.06,0.2,0.5,0.6], [0.65,0.07,0.14,0.86], [0.83,0.07,0.14,0.86]],
                       [None],
                       [[0.06,0.2,0.5,0.6], [0.65,0.07,0.14,0.86], [0.83,0.07,0.14,0.86]],
                       [[0.06,0.07,0.88,0.86]],
                       [[0.06,0.07,0.88,0.86]],
                       [[0.06,0.07,0.88,0.86]]]
        
        self.ax_sub_pos = [[[0.06,0.07,0.88,0.86]],
                           [[0.06,0.07,0.88,0.86], [0.06,0.07,0.88,0.86]],
                           [[0.06,0.07,0.88,0.86]],
                           [[0.06,0.07,0.88,0.86], [0.06,0.07,0.88,0.86]]]
        
        ax_s0 = self.fig.add_subplot(111)
        ax_s0.set_position([0.06,0.07,0.88,0.86])

        ax_s1_1 = self.fig.add_subplot(121)
        ax_s1_2 = self.fig.add_subplot(122, sharex=ax_s1_1, sharey=ax_s1_1)
        ax_s1_1.set_position([0.06,0.07,0.435,0.86])
        ax_s1_2.set_position([0.555,0.07,0.435,0.86])

        ax_s2_1 = self.fig.add_subplot(131)
        ax_s2_2 = self.fig.add_subplot(132)
        ax_s2_3 = self.fig.add_subplot(133, sharex=ax_s2_2, sharey=ax_s2_2)
        ax_s2_1.set_position([0.06,0.2,0.5,0.6])
        ax_s2_2.set_position([0.65,0.07,0.14,0.86])
        ax_s2_3.set_position([0.83,0.07,0.14,0.86])

        ax_s4_1 = self.fig.add_subplot(131)
        ax_s4_2 = self.fig.add_subplot(132)
        ax_s4_3 = self.fig.add_subplot(133, sharex=ax_s4_2, sharey=ax_s4_2)
        ax_s4_1.set_position([0.06,0.2,0.5,0.6])
        ax_s4_2.set_position([0.65,0.07,0.14,0.86])
        ax_s4_3.set_position([0.83,0.07,0.14,0.86])

        ax_s5 = self.fig.add_subplot(111)
        ax_s5.set_position([0.06,0.07,0.88,0.86])

        ax_s6 = self.fig.add_subplot(111)
        ax_s6.set_position([0.06,0.07,0.88,0.86])

        ax_s7_R1 = self.fig.add_subplot(111, title='-4 $\\AA$')
        ax_s7_R2 = self.fig.add_subplot(111, sharex=ax_s7_R1, sharey=ax_s7_R1, title='-0.5 $\\AA$')
        ax_s7_R3 = self.fig.add_subplot(111, sharex=ax_s7_R1, sharey=ax_s7_R1, xlabel='X (pix)', ylabel='Y (pix)', title='0.0 $\\AA$')
        ax_s7_R4 = self.fig.add_subplot(111, sharex=ax_s7_R1, sharey=ax_s7_R1, title='+0.5 $\\AA$')
        ax_s7_R1.set_axis_off()
        ax_s7_R2.set_axis_off()
        ax_s7_R3.set_axis_off()
        ax_s7_R4.set_axis_off()
        ax_s7_prof = self.fig.add_subplot(111, ylabel='Intensity (DN)', title='Profile')
        ax_s7_spec = self.fig.add_subplot(111, sharex=ax_s7_prof, xlabel='Wavelength (pix)', ylabel='Slit (pix)', title='Spectrogram')
        ax_s7_R1.set_position([0.06,0.54,0.17,0.4])
        ax_s7_R2.set_position([0.29,0.54,0.17,0.4])
        ax_s7_R3.set_position([0.06,0.07,0.17,0.4])
        ax_s7_R4.set_position([0.29,0.07,0.17,0.4])
        ax_s7_prof.set_position([0.56,0.57,0.38,0.36])
        ax_s7_spec.set_position([0.56,0.07,0.38,0.36])

        self.ax_s7_comp = self.fig.add_subplot(111)
        self.ax_s7_comp.set_position([0.06,0.07,0.88,0.86])
        self.ax_s7_comp.set_visible(False)

        self.ax = [[ax_s0], [ax_s1_1, ax_s1_2], [ax_s2_1, ax_s2_2, ax_s2_3], None, [ax_s4_1, ax_s4_2, ax_s4_3], [ax_s5], [ax_s6], [ax_s7_R1, ax_s7_R2, ax_s7_R3, ax_s7_R4, ax_s7_prof, ax_s7_spec]]
    

        ax_s3_1_1 = self.fig.add_subplot(111)
        ax_s3_1_1.set_position([0.06,0.07,0.88,0.86])

        ax_s3_2_1 = self.fig.add_subplot(111)
        ax_s3_2_2 = self.fig.add_subplot(111)
        ax_s3_2_1.set_position([0.06,0.07,0.88,0.86])
        ax_s3_2_2.set_position([0.06,0.07,0.88,0.86])

        ax_s3_3 = self.fig.add_subplot(111)
        ax_s3_3.set_position([0.06,0.07,0.88,0.86])

        ax_s3_4_1 = self.fig.add_subplot(111)
        ax_s3_4_2 = self.fig.add_subplot(111)
        ax_s3_4_1.set_position([0.06,0.07,0.88,0.86])
        ax_s3_4_2.set_position([0.06,0.07,0.88,0.86])

        self.ax_sub = [[ax_s3_1_1], [ax_s3_2_1, ax_s3_2_2], [ax_s3_3], [ax_s3_4_1, ax_s3_4_2]]

        self.ax_hide()
        for ax in self.ax[0]:
            ax.set_visible(True)
        self.initWidget()
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)
        self.s2 = None
        self.fig.show()
        qSleep(0.01)
        self.List_VL[0].setGeometry(QtCore.QRect(10,85,280,350))
        qSleep(0.01)

    def _onKey(self, event):
        if event.key == 'ctrl+c' or event.key == 'q':
            self.stop = True

    def ax_hide(self):
        for i, lax in enumerate(self.ax):
            if i != 3:
                for ax in lax:
                    ax.set_visible(False)
        for lax in self.ax_sub:
            for ax in lax:
                ax.set_visible(False)

    def step(self, num):
        h = self.fig.get_figheight()
        self.ax_hide()
        if self.stepNum !=3:
            for wg in self.StepWidgets[self.stepNum]:
                wg.setVisible(False)
        for ssw in self.subStepWidgets:
            for wg in ssw:
                wg.setVisible(False)

        for wg in self.StepWidgets[num]:
            wg.setVisible(True)

        
        self.L_Step.setText(self.List_step[num])
        self.log = f"<font color='{self.font_primary}'>"+self.List_step[num] + '</font><br><br>'
        self._writeLog()
        self.stepNum = num
        if num != 3:
            for ax in self.ax[num]:
                ax.set_visible(True)

        self.fig.canvas.draw_idle()

        
        if num == 1:
            if self.CF.ffoc is not None:
                self.foc = fits.getdata(self.CF.ffoc).mean(0)
                self.s1_data = self.foc
            else:
                self.s1_data = 10**self.CF.logRF[self.CF.nf//2]
        if num == 7:
            self.B_Next.setVisible(False)
            qSleep(0.05)
        else:
            self.B_Next.setVisible(True)
        
        self.List_VL[self.stepNum].setGeometry(QtCore.QRect(10,85,280,350))
        qSleep(0.05)
        self.fig.set_figheight(h)

    def subStep(self, num):
        h = self.fig.get_figheight()
        self.ax_hide()
        for wg in self.subStepWidgets[self.subStepNum]:
            wg.setVisible(False)
        for wg in self.subStepWidgets[num]:
            wg.setVisible(True)

        self.L_Step.setText(self.List_step[self.stepNum])
        self.log = f"<font color='{self.font_primary}'>"+self.List_step[self.stepNum] + '</font><br><br>'
        self.log += f"<font color='{self.font_second}'>"+self.List_subStep[num] + '</font><br><br>'
        self._writeLog()
        
        for ax in self.ax_sub[num]:
            ax.set_visible(True)
            if num == 1 or num == 3:
                break
        
        self.subStepNum = num
        self.fig.canvas.draw_idle()
        self.List_subVL[num].setGeometry(QtCore.QRect(10,85,280,350))
        self.fig.set_figheight(h)

    def initWidget(self):
        self.log = ""
        self.fNormal = QtGui.QFont()
        self.fNormal.setFamily("Arial")
        self.fNormal.setPointSize(12)
        self.fTitle = QtGui.QFont()
        self.fTitle.setFamily("Arial")
        self.fTitle.setPointSize(15)
        self.fTitle.setWeight(75)
        self.fMain = QtGui.QFont()
        self.fMain.setFamily("Arial")
        self.fMain.setPointSize(15)
        self.fMain.setWeight(75)

        # figure window to variable
        self.root = self.fig.canvas.manager.window
        self.dock = QtWidgets.QDockWidget("FISS Process Control Panel", self.root)
        
        # create pannel and layout
        self.panel = QtWidgets.QWidget()

        self.panel.setStyleSheet(f"background-color: {self.bg_primary}; color: {self.font_normal}; \n")
        h = self.fig.get_figheight()
        self.panel.setMaximumSize(QtCore.QSize(300, int(h*100)-22))
        self.panel.setMinimumSize(QtCore.QSize(300, 0))
        self.vboxAll = QtWidgets.QVBoxLayout(self.panel)
        self.vboxMain = QtWidgets.QVBoxLayout()
        self.vboxCtrl = QtWidgets.QVBoxLayout()


        # Title
        Title = QtWidgets.QLabel("<FISS Preprocess>")
        Title.setStyleSheet(f"color: {self.font_primary};")
        Title.setFont(self.fMain)
        Title.setAlignment(QtCore.Qt.AlignCenter)

        # Hline
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"color: {self.font_normal};")
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        line2.setStyleSheet(f"color: {self.font_normal};")
        line3 = QtWidgets.QFrame()
        line3.setFrameShape(QtWidgets.QFrame.HLine)
        line3.setStyleSheet(f"color: {self.font_normal};")

        # create step comboBox
        self.L_Step = QtWidgets.QLabel(self.List_step[0])
        self.L_Step.setFont(self.fTitle)
        self.L_Step.setStyleSheet(f"color: {self.font_second};")
        self.L_Step.setWordWrap(True)

        # add Widget in main vbox
        self.vboxMain.addWidget(Title)
        self.vboxMain.addWidget(line)
        self.vboxMain.addWidget(self.L_Step)
        # self.vboxMain.addWidget(self.CB_Step)

        # set step widgets
        self.StepWidgets = [None]*8#self.nStep
        self.subStepWidgets = [None]*4

        # create Step0 Widget
        if True:
            self.VL_s0 = QtWidgets.QVBoxLayout()
            self.L_s0_fflist = QtWidgets.QLabel()
            self.L_s0_fflist.setText("Flat file list")
            self.L_s0_fflist.setFont(self.fNormal)
            self.CB_s0_fflist = QtWidgets.QComboBox()
            self.CB_s0_fflist.setStyleSheet("background-color: %s; border: 1px solid %s;"%(self.bg_second, self.border))
            self.CB_s0_fflist.addItems(self.fflatGBL)
            self.CB_s0_fflist.setCurrentIndex(0)
            self.CB_s0_fflist.currentIndexChanged.connect(self.chFlat)

            # frame change
            self.HL_s0_frame = QtWidgets.QHBoxLayout()
            self.L_s0_frame = QtWidgets.QLabel()
            self.L_s0_frame.setText("Frame:")
            self.L_s0_frame.setFont(self.fNormal)
            self.LE_s0_frame = QtWidgets.QLineEdit()
            self.LE_s0_frame.setText("")
            self.LE_s0_frame.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s0_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.L_s0_nframe = QtWidgets.QLabel()
            self.L_s0_nframe.setText("/?")
            self.L_s0_nframe.setFont(self.fNormal)
            self.B_s0_pf = QtWidgets.QPushButton()
            self.B_s0_pf.setText("<")
            self.B_s0_pf.setFont(self.fNormal)
            self.B_s0_pf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s0_pf.clicked.connect(self.s0_pf)
            self.B_s0_nf = QtWidgets.QPushButton()
            self.B_s0_nf.setText(">")
            self.B_s0_nf.setFont(self.fNormal)
            self.B_s0_nf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s0_nf.clicked.connect(self.s0_nf)

            self.HL_s0_frame.addWidget(self.L_s0_frame)
            self.HL_s0_frame.addWidget(self.LE_s0_frame)
            self.HL_s0_frame.addWidget(self.L_s0_nframe)
            self.HL_s0_frame.addWidget(self.B_s0_pf)
            self.HL_s0_frame.addWidget(self.B_s0_nf)

            # go to preprocess step
            self.L_s0_step7 = QtWidgets.QLabel()
            self.L_s0_step7.setText("Go to Step7: preprocess")
            self.L_s0_step7.setFont(self.fNormal)
            self.B_s0_step7 = QtWidgets.QPushButton()
            self.B_s0_step7.setText("Go to Step7")
            self.B_s0_step7.setFont(self.fNormal)
            self.B_s0_step7.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s0_step7.clicked.connect(self.s0_step7)

            # add widgets
            self.VL_s0.addWidget(self.L_s0_fflist)
            self.VL_s0.addWidget(self.CB_s0_fflist)
            self.VL_s0.addLayout(self.HL_s0_frame)
            self.VL_s0.addWidget(self.L_s0_step7)
            self.VL_s0.addWidget(self.B_s0_step7)
            self.vboxCtrl.addLayout(self.VL_s0)

            self.StepWidgets[0] = [self.L_s0_fflist, self.CB_s0_fflist, self.L_s0_frame, self.LE_s0_frame, self.L_s0_nframe, self.B_s0_pf, self.B_s0_nf, self.L_s0_step7, self.B_s0_step7]     
            
        # create Step1 tilt Widget
        if True:
            self.VL_s1 = QtWidgets.QVBoxLayout()
            self.HL0_s1 = QtWidgets.QHBoxLayout()
            self.L_s1_get_tilt = QtWidgets.QLabel()
            self.L_s1_get_tilt.setText("Get Tilt: ")
            self.L_s1_get_tilt.setFont(self.fNormal)

            self.B_s1_run = QtWidgets.QPushButton()
            self.B_s1_run.setText("Calculate")
            self.B_s1_run.setFont(self.fNormal)
            self.B_s1_run.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            
            self.HL_s1 = QtWidgets.QHBoxLayout()
            self.L_s1_tilt = QtWidgets.QLabel()
            self.L_s1_tilt.setText("Tilt:")
            self.L_s1_tilt.setFont(self.fNormal)
            
            self.LE_s1_tilt = QtWidgets.QLineEdit()
            self.LE_s1_tilt.setText("0")
            self.LE_s1_tilt.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s1_tilt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            
            self.L_s1_deg = QtWidgets.QLabel()
            self.L_s1_deg.setText("deg")
            self.L_s1_deg.setFont(self.fNormal)

            self.B_s1_Apply = QtWidgets.QPushButton()
            self.B_s1_Apply.setText("Apply")
            self.B_s1_Apply.setFont(self.fNormal)
            self.B_s1_Apply.setStyleSheet(f"background-color: {self.bg_second};")
            
            self.HL0_s1.addWidget(self.L_s1_get_tilt)
            self.HL0_s1.addWidget(self.B_s1_run)

            self.HL_s1.addWidget(self.L_s1_tilt)
            self.HL_s1.addWidget(self.L_s1_tilt)
            self.HL_s1.addWidget(self.LE_s1_tilt)
            self.HL_s1.addWidget(self.L_s1_deg)
            self.HL_s1.addWidget(self.B_s1_Apply)

            self.VL_s1.addLayout(self.HL0_s1)
            self.VL_s1.addLayout(self.HL_s1)

            self.vboxCtrl.addLayout(self.VL_s1)

            self.StepWidgets[1] = [self.L_s1_get_tilt, self.B_s1_run, self.L_s1_tilt, self.LE_s1_tilt, self.L_s1_deg, self.B_s1_Apply]

            self.B_s1_Apply.setEnabled(False)

            self.B_s1_run.clicked.connect(self.s1_Run)
            self.B_s1_Apply.clicked.connect(self.s1_Apply)

        # create Step2 curvature Correction
        if True:
            self.VL_s2  = QtWidgets.QVBoxLayout()
            self.HL0_s2 = QtWidgets.QHBoxLayout()

            self.L_s2_get_curve = QtWidgets.QLabel()
            self.L_s2_get_curve.setText("Get curve coeff: ")
            self.L_s2_get_curve.setFont(self.fNormal)

            self.B_s2_run = QtWidgets.QPushButton()
            self.B_s2_run.setText("Calculate")
            self.B_s2_run.setFont(self.fNormal)
            self.B_s2_run.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")

            self.HL0_s2.addWidget(self.L_s2_get_curve)
            self.HL0_s2.addWidget(self.B_s2_run)


            self.L_s2_p0 = QtWidgets.QLabel()
            self.L_s2_p0.setText("p0: ")
            self.L_s2_p0.setFont(self.fNormal)

            self.LE_s2_p0 = QtWidgets.QLineEdit()
            self.LE_s2_p0.setText("0")
            self.LE_s2_p0.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s2_p0.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.L_s2_p1 = QtWidgets.QLabel()
            self.L_s2_p1.setText("p1: ")
            self.L_s2_p1.setFont(self.fNormal)

            self.LE_s2_p1 = QtWidgets.QLineEdit()
            self.LE_s2_p1.setText("0")
            self.LE_s2_p1.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s2_p1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.L_s2_p2 = QtWidgets.QLabel()
            self.L_s2_p2.setText("p2: ")
            self.L_s2_p2.setFont(self.fNormal)

            self.LE_s2_p2 = QtWidgets.QLineEdit()
            self.LE_s2_p2.setText("0")
            self.LE_s2_p2.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s2_p2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.B_s2_Apply = QtWidgets.QPushButton()
            self.B_s2_Apply.setText("Apply")
            self.B_s2_Apply.setFont(self.fNormal)
            self.B_s2_Apply.setStyleSheet(f"background-color: {self.bg_second};")

            self.HL_s2_p0 = QtWidgets.QHBoxLayout()
            self.HL_s2_p1 = QtWidgets.QHBoxLayout()
            self.HL_s2_p2 = QtWidgets.QHBoxLayout()
            
            self.HL_s2_p0.addWidget(self.L_s2_p0)
            self.HL_s2_p0.addWidget(self.LE_s2_p0)
            self.HL_s2_p1.addWidget(self.L_s2_p1)
            self.HL_s2_p1.addWidget(self.LE_s2_p1)
            self.HL_s2_p2.addWidget(self.L_s2_p2)
            self.HL_s2_p2.addWidget(self.LE_s2_p2)

            self.VL_s2.addLayout(self.HL0_s2)
            self.VL_s2.addLayout(self.HL_s2_p0)
            self.VL_s2.addLayout(self.HL_s2_p1)
            self.VL_s2.addLayout(self.HL_s2_p2)
            self.VL_s2.addWidget(self.B_s2_Apply)

            self.vboxCtrl.addLayout(self.VL_s2)

            self.StepWidgets[2] = [self.L_s2_get_curve, self.B_s2_run, self.L_s2_p0, self.LE_s2_p0, self.L_s2_p1, self.LE_s2_p1, self.L_s2_p2, self.LE_s2_p2, self.B_s2_Apply]

            self.B_s2_run.clicked.connect(self.s2_Run)
            self.B_s2_Apply.clicked.connect(self.s2_Apply)

            self.B_s2_Apply.setEnabled(False)

        # create Step3-1 atlas subtraction
        if True:
            self.VL_s3_1 = QtWidgets.QVBoxLayout()
            self.L_s3_1_subStep = QtWidgets.QLabel()
            self.L_s3_1_subStep.setText(self.List_subStep[0])
            self.L_s3_1_subStep.setFont(self.fNormal)
            self.L_s3_1_subStep.setStyleSheet(f"color: {self.font_third};")
            self.L_s3_1_subStep.setWordWrap(True)

            self.B_s3_1_run = QtWidgets.QPushButton()
            self.B_s3_1_run.setText("Run")
            self.B_s3_1_run.setFont(self.fNormal)
            self.B_s3_1_run.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")

            # frame change
            self.HL_s3_1_frame = QtWidgets.QHBoxLayout()
            self.L_s3_1_frame = QtWidgets.QLabel()
            self.L_s3_1_frame.setText("Frame:")
            self.L_s3_1_frame.setFont(self.fNormal)
            self.LE_s3_1_frame = QtWidgets.QLineEdit()
            self.LE_s3_1_frame.setText("")
            self.LE_s3_1_frame.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_1_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.L_s3_1_nframe = QtWidgets.QLabel()
            self.L_s3_1_nframe.setText("/?")
            self.L_s3_1_nframe.setFont(self.fNormal)
            self.B_s3_1_pf = QtWidgets.QPushButton()
            self.B_s3_1_pf.setText("<")
            self.B_s3_1_pf.setFont(self.fNormal)
            self.B_s3_1_pf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_1_pf.clicked.connect(self.s3_1_pf)
            self.B_s3_1_nf = QtWidgets.QPushButton()
            self.B_s3_1_nf.setText(">")
            self.B_s3_1_nf.setFont(self.fNormal)
            self.B_s3_1_nf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_1_nf.clicked.connect(self.s3_1_nf)

            self.HL_s3_1_frame.addWidget(self.L_s3_1_frame)
            self.HL_s3_1_frame.addWidget(self.LE_s3_1_frame)
            self.HL_s3_1_frame.addWidget(self.L_s3_1_nframe)
            self.HL_s3_1_frame.addWidget(self.B_s3_1_pf)
            self.HL_s3_1_frame.addWidget(self.B_s3_1_nf)

            self.VL_s3_1.addWidget(self.L_s3_1_subStep)
            self.VL_s3_1.addWidget(self.B_s3_1_run)
            self.VL_s3_1.addLayout(self.HL_s3_1_frame)

            self.vboxCtrl.addLayout(self.VL_s3_1)

            self.subStepWidgets[0] = [self.L_s3_1_subStep, self.B_s3_1_run, self.L_s3_1_frame, self.LE_s3_1_frame, self.L_s3_1_nframe, self.B_s3_1_pf, self.B_s3_1_nf]

            self.B_s3_1_run.clicked.connect(self.s3_1_Run)

        # create Step3-2 y Fringe
        if True:
            self.VL_s3_2 = QtWidgets.QVBoxLayout()
            self.L_s3_2_subStep = QtWidgets.QLabel()
            self.L_s3_2_subStep.setText(self.List_subStep[1])
            self.L_s3_2_subStep.setFont(self.fNormal)
            self.L_s3_2_subStep.setStyleSheet(f"color: {self.font_third};")
            self.L_s3_2_subStep.setWordWrap(True)

            # wavelet
            self.HL_wvlet = QtWidgets.QHBoxLayout()
            self.L_s3_2_wvlet = QtWidgets.QLabel()
            self.L_s3_2_wvlet.setText("Wavelet:")
            self.L_s3_2_wvlet.setFont(self.fNormal)
            self.B_s3_wvCal = QtWidgets.QPushButton()
            self.B_s3_wvCal.setText("Calculate")
            self.B_s3_wvCal.setFont(self.fNormal)
            self.B_s3_wvCal.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s3_wvCal.clicked.connect(self.s3_2_wvCal)

            self.B_s3_wvShow = QtWidgets.QPushButton()
            self.B_s3_wvShow.setText("Show")
            self.B_s3_wvShow.setFont(self.fNormal)
            self.B_s3_wvShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_wvShow.clicked.connect(self.s3_2_wvShow)

            self.HL_wvlet.addWidget(self.L_s3_2_wvlet)
            self.HL_wvlet.addWidget(self.B_s3_wvCal)
            self.HL_wvlet.addWidget(self.B_s3_wvShow)
            self.B_s3_wvShow.setEnabled(False)

            # Filter Range
            self.L_s3_2_FR = QtWidgets.QLabel()
            self.L_s3_2_FR.setText("Filter Range")
            self.L_s3_2_FR.setFont(self.fNormal)

            self.HL_FR = QtWidgets.QHBoxLayout()
            self.yf_min = 0
            self.L_s3_2_FRmin = QtWidgets.QLabel()
            self.L_s3_2_FRmin.setText("min:")
            self.L_s3_2_FRmin.setFont(self.fNormal)
            self.LE_s3_2_FRmin = QtWidgets.QLineEdit()
            self.LE_s3_2_FRmin.setText("0")
            self.LE_s3_2_FRmin.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_2_FRmin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.yf_max = 105
            self.L_s3_2_FRmax = QtWidgets.QLabel()
            self.L_s3_2_FRmax.setText("max:")
            self.L_s3_2_FRmax.setFont(self.fNormal)
            self.LE_s3_2_FRmax = QtWidgets.QLineEdit()
            self.LE_s3_2_FRmax.setText("105")
            self.LE_s3_2_FRmax.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_2_FRmax.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.B_s3_2_FRapply = QtWidgets.QPushButton()
            self.B_s3_2_FRapply.setText("Apply")
            self.B_s3_2_FRapply.setFont(self.fNormal)
            self.B_s3_2_FRapply.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_FRapply.clicked.connect(self.s3_2_FRapply)

            self.HL_FR.addWidget(self.L_s3_2_FRmin)
            self.HL_FR.addWidget(self.LE_s3_2_FRmin)
            self.HL_FR.addWidget(self.L_s3_2_FRmax)
            self.HL_FR.addWidget(self.LE_s3_2_FRmax)
            self.HL_FR.addWidget(self.B_s3_2_FRapply)
            self.B_s3_2_FRapply.setEnabled(False)

            # cal Fringe
            self.L_s3_2_calFringe = QtWidgets.QLabel()
            self.L_s3_2_calFringe.setText("Calculate Fringe:")
            self.L_s3_2_calFringe.setFont(self.fNormal)

            self.HL_Fringe = QtWidgets.QHBoxLayout()
            self.B_s3_2_simple = QtWidgets.QPushButton()
            self.B_s3_2_simple.setText("Calculate")
            self.B_s3_2_simple.setFont(self.fNormal)
            self.B_s3_2_simple.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s3_2_simple.clicked.connect(self.s3_2_simple)
            self.B_s3_2_simple.setEnabled(False)

            self.B_s3_2_FringeShow = QtWidgets.QPushButton()
            self.B_s3_2_FringeShow.setText("Show")
            self.B_s3_2_FringeShow.setFont(self.fNormal)
            self.B_s3_2_FringeShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_FringeShow.clicked.connect(self.s3_2_FringeShow)

            self.HL_Fringe.addWidget(self.B_s3_2_simple)
            self.HL_Fringe.addWidget(self.B_s3_2_FringeShow)
            self.B_s3_2_FringeShow.setEnabled(False)

            # show results
            self.HL_s3_2_res = QtWidgets.QHBoxLayout()
            self.L_s3_2_res = QtWidgets.QLabel()
            self.L_s3_2_res.setText("Fringe subtracted:")
            self.L_s3_2_res.setFont(self.fNormal)

            self.B_s3_2_resShow = QtWidgets.QPushButton()
            self.B_s3_2_resShow.setText("Show")
            self.B_s3_2_resShow.setFont(self.fNormal)
            self.B_s3_2_resShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_resShow.clicked.connect(self.s3_2_resShow)
            self.B_s3_2_resShow.setEnabled(False)

            self.B_s3_2_blink = QtWidgets.QPushButton()
            self.B_s3_2_blink.setText("Blink")
            self.B_s3_2_blink.setFont(self.fNormal)
            self.B_s3_2_blink.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_blink.clicked.connect(self.s3_2_blink)
            self.B_s3_2_blink.setEnabled(False)

            self.HL_s3_2_res.addWidget(self.B_s3_2_resShow)
            self.HL_s3_2_res.addWidget(self.B_s3_2_blink)

            # frame change
            self.HL_s3_2_frame = QtWidgets.QHBoxLayout()
            self.L_s3_2_frame = QtWidgets.QLabel()
            self.L_s3_2_frame.setText("Frame:")
            self.L_s3_2_frame.setFont(self.fNormal)
            self.LE_s3_2_frame = QtWidgets.QLineEdit()
            self.LE_s3_2_frame.setText("")
            self.LE_s3_2_frame.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_2_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.L_s3_2_nframe = QtWidgets.QLabel()
            self.L_s3_2_nframe.setText("/?")
            self.L_s3_2_nframe.setFont(self.fNormal)
            self.B_s3_2_pf = QtWidgets.QPushButton()
            self.B_s3_2_pf.setText("<")
            self.B_s3_2_pf.setFont(self.fNormal)
            self.B_s3_2_pf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_pf.clicked.connect(self.s3_2_pf)
            self.B_s3_2_nf = QtWidgets.QPushButton()
            self.B_s3_2_nf.setText(">")
            self.B_s3_2_nf.setFont(self.fNormal)
            self.B_s3_2_nf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_nf.clicked.connect(self.s3_2_nf)

            self.HL_s3_2_frame.addWidget(self.L_s3_2_frame)
            self.HL_s3_2_frame.addWidget(self.LE_s3_2_frame)
            self.HL_s3_2_frame.addWidget(self.L_s3_2_nframe)
            self.HL_s3_2_frame.addWidget(self.B_s3_2_pf)
            self.HL_s3_2_frame.addWidget(self.B_s3_2_nf)

            self.VL_s3_2.addWidget(self.L_s3_2_subStep)
            self.VL_s3_2.addLayout(self.HL_wvlet)
            self.VL_s3_2.addWidget(self.L_s3_2_FR)
            self.VL_s3_2.addLayout(self.HL_FR)
            self.VL_s3_2.addWidget(self.L_s3_2_calFringe)
            self.VL_s3_2.addLayout(self.HL_Fringe)
            self.VL_s3_2.addWidget(self.L_s3_2_res)
            self.VL_s3_2.addLayout(self.HL_s3_2_res)
            self.VL_s3_2.addLayout(self.HL_s3_2_frame)

            self.vboxCtrl.addLayout(self.VL_s3_2)

            self.subStepWidgets[1] = [self.L_s3_2_subStep, self.L_s3_2_wvlet, self.B_s3_wvCal, self.B_s3_wvShow, self.L_s3_2_FR, self.L_s3_2_FRmin, self.LE_s3_2_FRmin, self.L_s3_2_FRmax, self.LE_s3_2_FRmax, self.B_s3_2_FRapply, self.L_s3_2_calFringe, self.B_s3_2_simple, self.B_s3_2_FringeShow, self.L_s3_2_res, self.B_s3_2_resShow, self.B_s3_2_blink, self.L_s3_2_frame, self.LE_s3_2_frame, self.L_s3_2_nframe, self.B_s3_2_pf, self.B_s3_2_nf]
            self.imFy = None

        # create Step3-3 Data mask
        if True:
            self.VL_s3_3 = QtWidgets.QVBoxLayout()
            self.L_s3_3_subStep = QtWidgets.QLabel()
            self.L_s3_3_subStep.setText(self.List_subStep[2])
            self.L_s3_3_subStep.setFont(self.fNormal)
            self.L_s3_3_subStep.setStyleSheet(f"color: {self.font_third};")
            self.L_s3_3_subStep.setWordWrap(True)

            # get msk width
            self.HL_s3_3_GMW = QtWidgets.QHBoxLayout()
            self.L_s3_3_GMW = QtWidgets.QLabel()
            self.L_s3_3_GMW.setText("Masking: ")
            self.L_s3_3_GMW.setFont(self.fNormal)

            self.B_s3_3_run = QtWidgets.QPushButton()
            self.B_s3_3_run.setText("Run")
            self.B_s3_3_run.setFont(self.fNormal)
            self.B_s3_3_run.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s3_3_run.clicked.connect(self.s3_3_Run)

            self.HL_s3_3_GMW.addWidget(self.L_s3_3_GMW)
            self.HL_s3_3_GMW.addWidget(self.B_s3_3_run)

            # show image
            self.HL_s3_3_show = QtWidgets.QHBoxLayout()
            self.L_s3_3_res = QtWidgets.QLabel()
            self.L_s3_3_res.setText("Show results: ")
            self.L_s3_3_res.setFont(self.fNormal)


            self.B_s3_3_Blink = QtWidgets.QPushButton()
            self.B_s3_3_Blink.setText("Blink")
            self.B_s3_3_Blink.setFont(self.fNormal)
            self.B_s3_3_Blink.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_3_Blink.clicked.connect(self.s3_3_Blink)

            self.HL_s3_3_show.addWidget(self.L_s3_3_res)
            self.HL_s3_3_show.addWidget(self.B_s3_3_Blink)
            self.B_s3_3_Blink.setEnabled(False)

            # remove the mask
            self.HL_s3_3_reset = QtWidgets.QHBoxLayout()
            self.L_s3_3_reset = QtWidgets.QLabel()
            self.L_s3_3_reset.setText("Remove the mask?")
            self.L_s3_3_reset.setFont(self.fNormal)

            self.B_s3_3_reset = QtWidgets.QPushButton()
            self.B_s3_3_reset.setText("Remove")
            self.B_s3_3_reset.setFont(self.fNormal)
            self.B_s3_3_reset.setStyleSheet(f"background-color: {self.btn_2}; color:{self.bg_primary};")
            self.B_s3_3_reset.clicked.connect(self.s3_3_reset)

            self.HL_s3_3_reset.addWidget(self.L_s3_3_reset)
            self.HL_s3_3_reset.addWidget(self.B_s3_3_reset)

            # frame change
            self.HL_s3_3_frame = QtWidgets.QHBoxLayout()
            self.L_s3_3_frame = QtWidgets.QLabel()
            self.L_s3_3_frame.setText("Frame:")
            self.L_s3_3_frame.setFont(self.fNormal)
            self.LE_s3_3_frame = QtWidgets.QLineEdit()
            self.LE_s3_3_frame.setText("")
            self.LE_s3_3_frame.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_3_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.L_s3_3_nframe = QtWidgets.QLabel()
            self.L_s3_3_nframe.setText("/?")
            self.L_s3_3_nframe.setFont(self.fNormal)
            self.B_s3_3_pf = QtWidgets.QPushButton()
            self.B_s3_3_pf.setText("<")
            self.B_s3_3_pf.setFont(self.fNormal)
            self.B_s3_3_pf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_3_pf.clicked.connect(self.s3_3_pf)
            self.B_s3_3_nf = QtWidgets.QPushButton()
            self.B_s3_3_nf.setText(">")
            self.B_s3_3_nf.setFont(self.fNormal)
            self.B_s3_3_nf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_3_nf.clicked.connect(self.s3_3_nf)

            self.HL_s3_3_frame.addWidget(self.L_s3_3_frame)
            self.HL_s3_3_frame.addWidget(self.LE_s3_3_frame)
            self.HL_s3_3_frame.addWidget(self.L_s3_3_nframe)
            self.HL_s3_3_frame.addWidget(self.B_s3_3_pf)
            self.HL_s3_3_frame.addWidget(self.B_s3_3_nf)

            # addWidgets
            self.VL_s3_3.addWidget(self.L_s3_3_subStep)
            self.VL_s3_3.addLayout(self.HL_s3_3_GMW)
            self.VL_s3_3.addLayout(self.HL_s3_3_show)
            self.VL_s3_3.addLayout(self.HL_s3_3_reset)
            self.VL_s3_3.addLayout(self.HL_s3_3_frame)

            self.vboxCtrl.addLayout(self.VL_s3_3)
            self.subStepWidgets[2] = [self.L_s3_3_subStep, self.L_s3_3_GMW, self.B_s3_3_run, self.L_s3_3_res, self.B_s3_3_Blink, self.L_s3_3_frame, self.LE_s3_3_frame, self.L_s3_3_nframe, self.B_s3_3_pf, self.B_s3_3_nf, self.L_s3_3_reset, self.B_s3_3_reset]

            self.im_s3_3 = None
        
        # create Step3-4 x Fringe
        if True:
            self.VL_s3_4 = QtWidgets.QVBoxLayout()
            self.L_s3_4_subStep = QtWidgets.QLabel()
            self.L_s3_4_subStep.setText(self.List_subStep[3])
            self.L_s3_4_subStep.setFont(self.fNormal)
            self.L_s3_4_subStep.setStyleSheet(f"color: {self.font_third};")
            self.L_s3_4_subStep.setWordWrap(True)

            # wavelet
            self.HL_xwvlet = QtWidgets.QHBoxLayout()
            self.L_s3_4_wvlet = QtWidgets.QLabel()
            self.L_s3_4_wvlet.setText("Wavelet:")
            self.L_s3_4_wvlet.setFont(self.fNormal)
            self.B_s3_4_wvCal = QtWidgets.QPushButton()
            self.B_s3_4_wvCal.setText("Calculate")
            self.B_s3_4_wvCal.setFont(self.fNormal)
            self.B_s3_4_wvCal.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s3_4_wvCal.clicked.connect(self.s3_4_wvCal)

            self.B_s3_4_wvShow = QtWidgets.QPushButton()
            self.B_s3_4_wvShow.setText("Show")
            self.B_s3_4_wvShow.setFont(self.fNormal)
            self.B_s3_4_wvShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_wvShow.clicked.connect(self.s3_4_wvShow)

            self.HL_xwvlet.addWidget(self.L_s3_4_wvlet)
            self.HL_xwvlet.addWidget(self.B_s3_4_wvCal)
            self.HL_xwvlet.addWidget(self.B_s3_4_wvShow)
            self.B_s3_4_wvShow.setEnabled(False)

            # Filter Range
            self.L_s3_4_FR = QtWidgets.QLabel()
            self.L_s3_4_FR.setText("Filter Range")
            self.L_s3_4_FR.setFont(self.fNormal)

            self.HL_xFR = QtWidgets.QHBoxLayout()
            self.xf_min = 114
            self.L_s3_4_FRmin = QtWidgets.QLabel()
            self.L_s3_4_FRmin.setText("min:")
            self.L_s3_4_FRmin.setFont(self.fNormal)
            self.LE_s3_4_FRmin = QtWidgets.QLineEdit()
            self.LE_s3_4_FRmin.setText("114")
            self.LE_s3_4_FRmin.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_4_FRmin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.xf_max = 124
            self.L_s3_4_FRmax = QtWidgets.QLabel()
            self.L_s3_4_FRmax.setText("max:")
            self.L_s3_4_FRmax.setFont(self.fNormal)
            self.LE_s3_4_FRmax = QtWidgets.QLineEdit()
            self.LE_s3_4_FRmax.setText("124")
            self.LE_s3_4_FRmax.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_4_FRmax.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.B_s3_4_FRapply = QtWidgets.QPushButton()
            self.B_s3_4_FRapply.setText("Apply")
            self.B_s3_4_FRapply.setFont(self.fNormal)
            self.B_s3_4_FRapply.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_FRapply.clicked.connect(self.s3_4_FRapply)

            self.HL_xFR.addWidget(self.L_s3_4_FRmin)
            self.HL_xFR.addWidget(self.LE_s3_4_FRmin)
            self.HL_xFR.addWidget(self.L_s3_4_FRmax)
            self.HL_xFR.addWidget(self.LE_s3_4_FRmax)
            self.HL_xFR.addWidget(self.B_s3_4_FRapply)
            self.B_s3_4_FRapply.setEnabled(False)

            # cal Fringe
            self.L_s3_4_calFringe = QtWidgets.QLabel()
            self.L_s3_4_calFringe.setText("Calculate Fringe:")
            self.L_s3_4_calFringe.setFont(self.fNormal)

            self.HL_xFringe = QtWidgets.QHBoxLayout()
            self.B_s3_4_simple = QtWidgets.QPushButton()
            self.B_s3_4_simple.setText("Cal (simple)")
            self.B_s3_4_simple.setFont(self.fNormal)
            self.B_s3_4_simple.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s3_4_simple.clicked.connect(self.s3_4_simple)
            self.B_s3_4_simple.setEnabled(False)

            self.B_s3_4_gauss = QtWidgets.QPushButton()
            self.B_s3_4_gauss.setText("Cal (Gauss)")
            self.B_s3_4_gauss.setFont(self.fNormal)
            self.B_s3_4_gauss.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_gauss.clicked.connect(self.s3_4_gauss)
            self.B_s3_4_gauss.setEnabled(False)

            self.B_s3_4_FringeShow = QtWidgets.QPushButton()
            self.B_s3_4_FringeShow.setText("Show")
            self.B_s3_4_FringeShow.setFont(self.fNormal)
            self.B_s3_4_FringeShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_FringeShow.clicked.connect(self.s3_4_FringeShow)

            self.HL_xFringe.addWidget(self.B_s3_4_simple)
            self.HL_xFringe.addWidget(self.B_s3_4_gauss)
            self.HL_xFringe.addWidget(self.B_s3_4_FringeShow)
            self.B_s3_4_FringeShow.setEnabled(False)

            # show results
            self.HL_s3_4_res = QtWidgets.QHBoxLayout()
            self.L_s3_4_res = QtWidgets.QLabel()
            self.L_s3_4_res.setText("Fringe subtracted:")
            self.L_s3_4_res.setFont(self.fNormal)

            self.B_s3_4_resShow = QtWidgets.QPushButton()
            self.B_s3_4_resShow.setText("Show")
            self.B_s3_4_resShow.setFont(self.fNormal)
            self.B_s3_4_resShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_resShow.clicked.connect(self.s3_4_resShow)
            self.B_s3_4_resShow.setEnabled(False)

            self.B_s3_4_blink = QtWidgets.QPushButton()
            self.B_s3_4_blink.setText("Blink")
            self.B_s3_4_blink.setFont(self.fNormal)
            self.B_s3_4_blink.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_blink.clicked.connect(self.s3_4_blink)
            self.B_s3_4_blink.setEnabled(False)

            self.HL_s3_4_res.addWidget(self.B_s3_4_resShow)
            self.HL_s3_4_res.addWidget(self.B_s3_4_blink)


            # reset
            self.HL_s3_4_reset = QtWidgets.QHBoxLayout()
            self.L_s3_4_reset = QtWidgets.QLabel()
            self.L_s3_4_reset.setText("Reset?")
            self.L_s3_4_reset.setFont(self.fNormal)

            self.B_s3_4_reset = QtWidgets.QPushButton()
            self.B_s3_4_reset.setText("Reset")
            self.B_s3_4_reset.setFont(self.fNormal)
            self.B_s3_4_reset.setStyleSheet(f"background-color: {self.btn_2}; color:{self.bg_primary};")
            self.B_s3_4_reset.clicked.connect(self.s3_4_reset)

            self.HL_s3_4_reset.addWidget(self.L_s3_4_reset)
            self.HL_s3_4_reset.addWidget(self.B_s3_4_reset)

            # frame change
            self.HL_s3_4_frame = QtWidgets.QHBoxLayout()
            self.L_s3_4_frame = QtWidgets.QLabel()
            self.L_s3_4_frame.setText("Frame:")
            self.L_s3_4_frame.setFont(self.fNormal)
            self.LE_s3_4_frame = QtWidgets.QLineEdit()
            self.LE_s3_4_frame.setText("")
            self.LE_s3_4_frame.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_4_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.L_s3_4_nframe = QtWidgets.QLabel()
            self.L_s3_4_nframe.setText("/?")
            self.L_s3_4_nframe.setFont(self.fNormal)
            self.B_s3_4_pf = QtWidgets.QPushButton()
            self.B_s3_4_pf.setText("<")
            self.B_s3_4_pf.setFont(self.fNormal)
            self.B_s3_4_pf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_pf.clicked.connect(self.s3_4_pf)
            self.B_s3_4_nf = QtWidgets.QPushButton()
            self.B_s3_4_nf.setText(">")
            self.B_s3_4_nf.setFont(self.fNormal)
            self.B_s3_4_nf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_4_nf.clicked.connect(self.s3_4_nf)

            self.HL_s3_4_frame.addWidget(self.L_s3_4_frame)
            self.HL_s3_4_frame.addWidget(self.LE_s3_4_frame)
            self.HL_s3_4_frame.addWidget(self.L_s3_4_nframe)
            self.HL_s3_4_frame.addWidget(self.B_s3_4_pf)
            self.HL_s3_4_frame.addWidget(self.B_s3_4_nf)

            self.VL_s3_4.addWidget(self.L_s3_4_subStep)
            self.VL_s3_4.addLayout(self.HL_xwvlet)
            self.VL_s3_4.addWidget(self.L_s3_4_FR)
            self.VL_s3_4.addLayout(self.HL_xFR)
            self.VL_s3_4.addWidget(self.L_s3_4_calFringe)
            self.VL_s3_4.addLayout(self.HL_xFringe)
            self.VL_s3_4.addWidget(self.L_s3_4_res)
            self.VL_s3_4.addLayout(self.HL_s3_4_res)
            self.VL_s3_4.addLayout(self.HL_s3_4_reset)
            self.VL_s3_4.addLayout(self.HL_s3_4_frame)

            self.vboxCtrl.addLayout(self.VL_s3_4)

            self.subStepWidgets[3] = [self.L_s3_4_subStep, self.L_s3_4_wvlet, self.B_s3_4_wvCal, self.B_s3_4_wvShow, self.L_s3_4_FR, self.L_s3_4_FRmin, self.LE_s3_4_FRmin, self.L_s3_4_FRmax, self.LE_s3_4_FRmax, self.B_s3_4_FRapply, self.L_s3_4_calFringe, self.B_s3_4_simple, self.B_s3_4_gauss, self.B_s3_4_FringeShow, self.L_s3_4_res, self.B_s3_4_resShow, self.B_s3_4_blink, self.L_s3_4_frame, self.LE_s3_4_frame, self.L_s3_4_nframe, self.B_s3_4_pf, self.B_s3_4_nf, self.L_s3_4_reset, self.B_s3_4_reset]
            self.imFx = None

        # create Step4 2nd curvature correciton
        if True:
            self.VL_s4  = QtWidgets.QVBoxLayout()
            self.HL0_s4 = QtWidgets.QHBoxLayout()

            self.L_s4_get_curve = QtWidgets.QLabel()
            self.L_s4_get_curve.setText("Get curve coeff: ")
            self.L_s4_get_curve.setFont(self.fNormal)

            self.B_s4_run = QtWidgets.QPushButton()
            self.B_s4_run.setText("Calculate")
            self.B_s4_run.setFont(self.fNormal)
            self.B_s4_run.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")

            self.HL0_s4.addWidget(self.L_s4_get_curve)
            self.HL0_s4.addWidget(self.B_s4_run)


            self.L_s4_p0 = QtWidgets.QLabel()
            self.L_s4_p0.setText("p0: ")
            self.L_s4_p0.setFont(self.fNormal)

            self.LE_s4_p0 = QtWidgets.QLineEdit()
            self.LE_s4_p0.setText("0")
            self.LE_s4_p0.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s4_p0.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.L_s4_p1 = QtWidgets.QLabel()
            self.L_s4_p1.setText("p1: ")
            self.L_s4_p1.setFont(self.fNormal)

            self.LE_s4_p1 = QtWidgets.QLineEdit()
            self.LE_s4_p1.setText("0")
            self.LE_s4_p1.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s4_p1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.L_s4_p2 = QtWidgets.QLabel()
            self.L_s4_p2.setText("p2: ")
            self.L_s4_p2.setFont(self.fNormal)

            self.LE_s4_p2 = QtWidgets.QLineEdit()
            self.LE_s4_p2.setText("0")
            self.LE_s4_p2.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s4_p2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.B_s4_Apply = QtWidgets.QPushButton()
            self.B_s4_Apply.setText("Apply")
            self.B_s4_Apply.setFont(self.fNormal)
            self.B_s4_Apply.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s4_Apply.setEnabled(False)

            self.HL_s4_p0 = QtWidgets.QHBoxLayout()
            self.HL_s4_p1 = QtWidgets.QHBoxLayout()
            self.HL_s4_p2 = QtWidgets.QHBoxLayout()
            
            self.HL_s4_p0.addWidget(self.L_s4_p0)
            self.HL_s4_p0.addWidget(self.LE_s4_p0)
            self.HL_s4_p1.addWidget(self.L_s4_p1)
            self.HL_s4_p1.addWidget(self.LE_s4_p1)
            self.HL_s4_p2.addWidget(self.L_s4_p2)
            self.HL_s4_p2.addWidget(self.LE_s4_p2)

            self.VL_s4.addLayout(self.HL0_s4)
            self.VL_s4.addLayout(self.HL_s4_p0)
            self.VL_s4.addLayout(self.HL_s4_p1)
            self.VL_s4.addLayout(self.HL_s4_p2)
            self.VL_s4.addWidget(self.B_s4_Apply)

            self.vboxCtrl.addLayout(self.VL_s4)

            self.StepWidgets[4] = [self.L_s4_get_curve, self.B_s4_run, self.L_s4_p0, self.LE_s4_p0, self.L_s4_p1, self.LE_s4_p1, self.L_s4_p2, self.LE_s4_p2, self.B_s4_Apply]

            self.B_s4_run.clicked.connect(self.s4_Run)
            self.B_s4_Apply.clicked.connect(self.s4_Apply)

        # create Step5 makeFlat
        if True:
            self.VL_s5  = QtWidgets.QVBoxLayout()
            self.HL_s5  = QtWidgets.QHBoxLayout()

            self.L_s5_label = QtWidgets.QLabel()
            self.L_s5_label.setText("Make Flat: ")
            self.L_s5_label.setFont(self.fNormal)
            self.B_s5_Run = QtWidgets.QPushButton()
            self.B_s5_Run.setText("Run")
            self.B_s5_Run.setFont(self.fNormal)
            self.B_s5_Run.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s5_Run.clicked.connect(self.s5_Run)
            self.B_s5_Show = QtWidgets.QPushButton()
            self.B_s5_Show.setText("Show")
            self.B_s5_Show.setFont(self.fNormal)
            self.B_s5_Show.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_Show.clicked.connect(self.s5_Show)
            
            self.HL_s5.addWidget(self.L_s5_label)
            self.HL_s5.addWidget(self.B_s5_Run)
            self.HL_s5.addWidget(self.B_s5_Show)
            self.B_s5_Show.setEnabled(False)

            self.HL_s5_cFlat = QtWidgets.QHBoxLayout()
            self.L_s5_cFlat = QtWidgets.QLabel()
            self.L_s5_cFlat.setText("Flat-fielded Flat: ")
            self.L_s5_cFlat.setFont(self.fNormal)
            self.B_s5_cFlat = QtWidgets.QPushButton()
            self.B_s5_cFlat.setText("Show")
            self.B_s5_cFlat.setFont(self.fNormal)
            self.B_s5_cFlat.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_cFlat.clicked.connect(self.s5_cFlat)

            self.B_s5_Blink = QtWidgets.QPushButton()
            self.B_s5_Blink.setText("Blink")
            self.B_s5_Blink.setFont(self.fNormal)
            self.B_s5_Blink.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_Blink.clicked.connect(self.s5_Blink)

            self.HL_s5_cFlat.addWidget(self.L_s5_cFlat)
            self.HL_s5_cFlat.addWidget(self.B_s5_cFlat)
            self.HL_s5_cFlat.addWidget(self.B_s5_Blink)
            self.B_s5_cFlat.setEnabled(False)
            self.B_s5_Blink.setEnabled(False)

            self.HL_s5_msFlat = QtWidgets.QHBoxLayout()
            self.L_s5_msFlat = QtWidgets.QLabel()
            self.L_s5_msFlat.setText("mean subtracted: ")
            self.L_s5_msFlat.setFont(self.fNormal)
            self.B_s5_msFlat = QtWidgets.QPushButton()
            self.B_s5_msFlat.setText("Show")
            self.B_s5_msFlat.setFont(self.fNormal)
            self.B_s5_msFlat.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_msFlat.clicked.connect(self.s5_msShow)

            self.HL_s5_msFlat.addWidget(self.L_s5_msFlat)
            self.HL_s5_msFlat.addWidget(self.B_s5_msFlat)
            self.B_s5_msFlat.setEnabled(False)

            self.HL_s5_profile = QtWidgets.QHBoxLayout()
            self.L_s5_profile = QtWidgets.QLabel()
            self.L_s5_profile.setText("Profile: ")
            self.L_s5_profile.setFont(self.fNormal)
            self.B_s5_profile = QtWidgets.QPushButton()
            self.B_s5_profile.setText("Show")
            self.B_s5_profile.setFont(self.fNormal)
            self.B_s5_profile.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_profile.clicked.connect(self.s5_profShow)

            self.HL_s5_profile.addWidget(self.L_s5_profile)
            self.HL_s5_profile.addWidget(self.B_s5_profile)
            self.p_s5_ori = None
            self.p_s5_ff = None
            self.im_s5 = None


             # frame change
            self.HL_s5_frame = QtWidgets.QHBoxLayout()
            self.L_s5_frame = QtWidgets.QLabel()
            self.L_s5_frame.setText("Frame:")
            self.L_s5_frame.setFont(self.fNormal)
            self.LE_s5_frame = QtWidgets.QLineEdit()
            self.LE_s5_frame.setText("")
            self.LE_s5_frame.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s5_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
            self.L_s5_nframe = QtWidgets.QLabel()
            self.L_s5_nframe.setText("/?")
            self.L_s5_nframe.setFont(self.fNormal)
            self.B_s5_pf = QtWidgets.QPushButton()
            self.B_s5_pf.setText("<")
            self.B_s5_pf.setFont(self.fNormal)
            self.B_s5_pf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_pf.clicked.connect(self.s5_pf)
            self.B_s5_nf = QtWidgets.QPushButton()
            self.B_s5_nf.setText(">")
            self.B_s5_nf.setFont(self.fNormal)
            self.B_s5_nf.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s5_nf.clicked.connect(self.s5_nf)

            self.HL_s5_frame.addWidget(self.L_s5_frame)
            self.HL_s5_frame.addWidget(self.LE_s5_frame)
            self.HL_s5_frame.addWidget(self.L_s5_nframe)
            self.HL_s5_frame.addWidget(self.B_s5_pf)
            self.HL_s5_frame.addWidget(self.B_s5_nf)
            
            self.VL_s5.addLayout(self.HL_s5)
            self.VL_s5.addLayout(self.HL_s5_cFlat)
            self.VL_s5.addLayout(self.HL_s5_msFlat)
            self.VL_s5.addLayout(self.HL_s5_profile)
            self.VL_s5.addLayout(self.HL_s5_frame)

            self.vboxCtrl.addLayout(self.VL_s5)

            self.StepWidgets[5] = [self.L_s5_label, self.B_s5_Run, self.B_s5_Show, self.L_s5_cFlat, self.B_s5_cFlat, self.B_s5_Blink, self.L_s5_msFlat, self.B_s5_msFlat, self.L_s5_profile, self.B_s5_profile, self.L_s5_frame, self.LE_s5_frame, self.L_s5_nframe, self.B_s5_pf, self.B_s5_nf]

        # create Step6 Save Flat
        if True:
            self.VL_s6 = QtWidgets.QVBoxLayout()
            self.HL_s6 = QtWidgets.QHBoxLayout()

            self.L_s6_save = QtWidgets.QLabel()
            self.L_s6_save.setText("Save Flat: ")
            self.L_s6_save.setFont(self.fNormal)

            self.B_s6_save = QtWidgets.QPushButton()
            self.B_s6_save.setText("Save")
            self.B_s6_save.setFont(self.fNormal)
            self.B_s6_save.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s6_save.clicked.connect(self.s6_save)

            self.HL_s6.addWidget(self.L_s6_save)
            self.HL_s6.addWidget(self.B_s6_save)

            self.L_s6_ask = QtWidgets.QLabel()
            self.L_s6_ask.setText("Are there any flat Files?")
            self.L_s6_ask.setFont(self.fNormal)

            self.B_s6_yes = QtWidgets.QPushButton()
            self.B_s6_yes.setText("Yes (go to Step0)")
            self.B_s6_yes.setFont(self.fNormal)
            self.B_s6_yes.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s6_yes.clicked.connect(self.s6_yes)

            self.B_s6_No = QtWidgets.QPushButton()
            self.B_s6_No.setText("No (Next Step)")
            self.B_s6_No.setFont(self.fNormal)
            self.B_s6_No.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s6_No.clicked.connect(self.Next)

            self.VL_s6.addLayout(self.HL_s6)
            self.VL_s6.addWidget(self.L_s6_ask)
            self.VL_s6.addWidget(self.B_s6_yes)
            self.VL_s6.addWidget(self.B_s6_No)

            self.vboxCtrl.addLayout(self.VL_s6)

            self.StepWidgets[6] = [self.L_s6_save, self.B_s6_save, self.L_s6_ask, self.B_s6_yes, self.B_s6_No]

        # create Step7 Run Preprocess
        if True:
            self.VL_s7 = QtWidgets.QVBoxLayout()

            self.HL_s7_cam = QtWidgets.QHBoxLayout()
            self.L_s7_cam = QtWidgets.QLabel()
            self.L_s7_cam.setText("Select Camera:")
            self.L_s7_cam.setFont(self.fNormal)

            self.CB_s7_camlist = QtWidgets.QComboBox()
            self.CB_s7_camlist.setStyleSheet("background-color: %s; border: 1px solid %s;"%(self.bg_second, self.border))
            self.CB_s7_camlist.addItems(['0: both','1: A','2: B'])
            self.CB_s7_camlist.setCurrentIndex(self.runCam)

            self.HL_s7_cam.addWidget(self.L_s7_cam)
            self.HL_s7_cam.addWidget(self.CB_s7_camlist)

            self.HL_s7_target = QtWidgets.QHBoxLayout()
            self.L_s7_target = QtWidgets.QLabel()
            self.L_s7_target.setText("Select Target:")
            self.L_s7_target.setFont(self.fNormal)

            self.CB_s7_targetlist = QtWidgets.QComboBox()
            self.CB_s7_targetlist.setStyleSheet("background-color: %s; border: 1px solid %s;"%(self.bg_second, self.border))
            lTarget = glob(join(self.rawdir, '*'))
            lTarget.sort()
            target = [None] * (len(lTarget)+1)
            target[0] = '0: All'
            for ii, t in enumerate(lTarget):
                target[ii+1] = f"{ii+1}: {basename(t)}"
            self.CB_s7_targetlist.addItems(target)
            self.CB_s7_targetlist.setCurrentIndex(self.runCam)

            self.HL_s7_target.addWidget(self.L_s7_target)
            self.HL_s7_target.addWidget(self.CB_s7_targetlist)

            self.L_s7_proc = QtWidgets.QLabel()
            self.L_s7_proc.setText("Run Preprocess:")
            self.L_s7_proc.setFont(self.fNormal)

            self.B_s7_proc = QtWidgets.QPushButton()
            self.B_s7_proc.setText("Run")
            self.B_s7_proc.setFont(self.fNormal)
            self.B_s7_proc.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s7_proc.clicked.connect(self.s7_proc)

            self.L_s7_comp = QtWidgets.QLabel()
            self.L_s7_comp.setText("Run PCA compression:")
            self.L_s7_comp.setFont(self.fNormal)
            
            self.B_s7_comp = QtWidgets.QPushButton()
            self.B_s7_comp.setText("Run")
            self.B_s7_comp.setFont(self.fNormal)
            self.B_s7_comp.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
            self.B_s7_comp.clicked.connect(self.s7_comp)

            self.L_s7_stop = QtWidgets.QLabel()
            self.L_s7_stop.setText("Stop")
            self.L_s7_stop.setFont(self.fNormal)
            
            self.B_s7_stop = QtWidgets.QPushButton()
            self.B_s7_stop.setText("Stop")
            self.B_s7_stop.setFont(self.fNormal)
            self.B_s7_stop.setStyleSheet(f"background-color: {self.font_err}; color: {self.bg_primary};")
            self.B_s7_stop.clicked.connect(self.s7_stop)

            self.VL_s7.addLayout(self.HL_s7_cam)
            self.VL_s7.addLayout(self.HL_s7_target)
            self.VL_s7.addWidget(self.L_s7_proc)
            self.VL_s7.addWidget(self.B_s7_proc)
            self.VL_s7.addWidget(self.L_s7_comp)
            self.VL_s7.addWidget(self.B_s7_comp)
            self.VL_s7.addWidget(self.L_s7_stop)
            self.VL_s7.addWidget(self.B_s7_stop)
            self.vboxCtrl.addLayout(self.VL_s7)

            self.StepWidgets[7] = [self.L_s7_cam, self.CB_s7_camlist, self.L_s7_target, self.CB_s7_targetlist, self.L_s7_proc, self.B_s7_proc, self.L_s7_comp, self.B_s7_comp, self.L_s7_stop, self.B_s7_stop]


        self.List_VL = [self.VL_s0, self.VL_s1, self.VL_s2, None, self.VL_s4, self.VL_s5, self.VL_s6, self.VL_s7]
        self.List_subVL = [self.VL_s3_1, self.VL_s3_2, self.VL_s3_3, self.VL_s3_4]

        for i,vl in enumerate(self.List_VL):
            if i != 3:
                vl.setAlignment(QtCore.Qt.AlignTop)
        for vl in self.List_subVL:
            vl.setAlignment(QtCore.Qt.AlignTop)
        # add Widget in control vbox
        for i, SW in enumerate(self.StepWidgets):
            if i !=3:
                for wg in SW:
                    if i != 0:
                        wg.setVisible(False)
        for i, SW in enumerate(self.subStepWidgets):
            for wg in SW:
                wg.setVisible(False)
            
        
        # vertical spcaer
        vSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        vSpacerInner = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # self.vboxCtrl.addItem(vSpacerInner)

        # prev next button
        if True:
            self.HL_PN = QtWidgets.QHBoxLayout()

            self.B_Prev = QtWidgets.QPushButton()
            self.B_Prev.setText("Prev")
            self.B_Prev.setFont(self.fNormal)
            self.B_Prev.setStyleSheet(f"background-color: {self.bg_second};")

            self.B_Next = QtWidgets.QPushButton()
            self.B_Next.setText("Next")
            self.B_Next.setFont(self.fNormal)
            self.B_Next.setStyleSheet(f"background-color: {self.bg_second};")

            self.HL_PN.addWidget(self.B_Prev)
            self.HL_PN.addWidget(self.B_Next)
            self.B_Next.clicked.connect(self.Next)
            self.B_Prev.clicked.connect(self.Prev)

        # Status
        if True:
            self.G_Log = QtWidgets.QGroupBox()
            self.G_Log.setMinimumSize(QtCore.QSize(100, 200))
            self.G_Log.setMaximumSize(QtCore.QSize(300, 200))
            self.G_Log.setFont(self.fTitle)
            self.G_Log.setFlat(True)
            self.G_Log.setTitle("Log")
            self.G_Log.setStyleSheet(f"color: {self.font_third};")
            self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.G_Log)
            self.horizontalLayout_2.setContentsMargins(0, 10, 0, 10)
            self.horizontalLayout_2.setSpacing(0)
            self.horizontalLayout_2.setObjectName("horizontalLayout_2")
            self.scrollArea = QtWidgets.QScrollArea(self.G_Log)
            self.scrollArea.setStyleSheet("")
            self.scrollArea.setFrameShadow(QtWidgets.QFrame.Sunken)
            # self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setObjectName("scrollArea")
            self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
            self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 300, 300))
            self.scrollAreaWidgetContents_2.setStyleSheet(f"background-color: {self.bg_second};")
            self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
            self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_2)
            self.horizontalLayout.setObjectName("horizontalLayout")
            self.L_Log = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
            self.L_Log.setStyleSheet("")
            self.L_Log.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
            self.L_Log.setObjectName("L_Log")
            self.horizontalLayout.addWidget(self.L_Log)
            self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
            self.horizontalLayout_2.addWidget(self.scrollArea)
            self.L_Log.setWordWrap(True)
            self.L_Log.setFont(self.fNormal)
            self.L_Log.setStyleSheet(f"color: {self.font_normal};")
            self.L_Log.setAlignment(QtCore.Qt.AlignTop)
        
        self.log = f"<font color='{self.font_primary}'>"+self.List_step[0] + '</font><br><br>'
        self._writeLog()
        self.chFlat()
        self.LE_s0_frame.textChanged.connect(self._txtCH_LE_s0)

        self.LE_s3_1_frame.setText(f"{self.frameNum+1}")
        self.L_s3_1_nframe.setText(f"/{self.CF.nf}")
        self.LE_s3_1_frame.textChanged.connect(self._txtCH_LE_s3_1)

        self.LE_s3_2_frame.setText(f"{self.frameNum+1}")
        self.L_s3_2_nframe.setText(f"/{self.CF.nf}")
        self.LE_s3_2_frame.textChanged.connect(self._txtCH_LE_s3_2)

        self.LE_s3_3_frame.setText(f"{self.frameNum+1}")
        self.L_s3_3_nframe.setText(f"/{self.CF.nf}")
        self.LE_s3_3_frame.textChanged.connect(self._txtCH_LE_s3_3)

        self.LE_s3_4_frame.setText(f"{self.frameNum+1}")
        self.L_s3_4_nframe.setText(f"/{self.CF.nf}")
        self.LE_s3_4_frame.textChanged.connect(self._txtCH_LE_s3_4)

        self.LE_s5_frame.setText(f"{self.frameNum+1}")
        self.L_s5_nframe.setText(f"/{self.CF.nf}")
        self.LE_s5_frame.textChanged.connect(self._txtCH_LE_s5)

        # add layout
        self.vboxAll.addLayout(self.vboxMain)
        self.vboxAll.addLayout(self.vboxCtrl)
        self.vboxAll.addItem(vSpacer)
        self.vboxAll.addWidget(line2)
        self.vboxAll.addLayout(self.HL_PN)
        self.vboxAll.addWidget(self.G_Log)
        

        self.vboxAll.setContentsMargins(10,10,10,0)
        
        
        self.dock.setWidget(self.panel)
        self.root.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock)

        
    def chFlat(self):
        self.initS1 = True
        self.initS2 = True
        self.initS3_1 = True
        self.initS3_2 = True
        self.initS3_3 = True
        self.initS3_4 = True
        self.initS4 = True
        self.initS5 = True
        self.fidx = self.CB_s0_fflist.currentIndex()
        f = self.fflatL[self.fidx]
        n = f.find('A_Flat')
        if n != -1:
            ffoc = self.ffocA
            self.bandA = True
        else:
            ffoc = self.ffocB
            self.bandA = False
        self.CF = proc_base.calFlat(f, ffoc)
        self.frameNum = self.CF.nf//2

        # draw image
        
        self.ax[0][0].cla()
        self.im_s0 = self.ax[0][0].imshow(self.CF.logRF[self.frameNum,5:-5,5:-5], plt.cm.gray, origin='lower', interpolation='nearest')
        self.ax[0][0].set_xlabel('Wavelength (pix)')
        self.ax[0][0].set_ylabel('Slit (pix)')
        self.ax[0][0].set_title(f'Raw Flat {self.CF.date}')
        self.fig.canvas.draw_idle()

        self.log += f"> Read Flat: {self.fflatGBL[self.fidx]}<br>"
        self._writeLog()
        self.LE_s0_frame.setText(f"{self.frameNum+1}")
        self.L_s0_nframe.setText(f"/{self.CF.nf}")

    def _writeLog(self):
        self.L_Log.setText(self.log)
        qSleep(0.01)
        # self.fig.canvas.draw_idle()
        self._scrollDown()

    def _scrollDown(self):
        sbar = self.scrollArea.verticalScrollBar()
        sbar.setValue(sbar.maximum())

    def Next(self):
        if self.stepNum == 2:
            for wg in self.StepWidgets[self.stepNum]:
                wg.setVisible(False)
            self.stepNum += 1
            self.subStep(self.subStepNum+1)
        elif self.stepNum == 3 and self.subStepNum != self.nsubStep-1:
            if self.bandA and (self.subStepNum == 1 or self.subStepNum == 2):
                self.subStepNum = 3
                self.step(self.stepNum+1)
            else:
                self.subStep(self.subStepNum+1)
        elif self.stepNum == 3 and self.subStepNum == self.nsubStep-1:
            for wg in self.subStepWidgets[self.subStepNum]:
                wg.setVisible(False)
            self.step(self.stepNum+1)
        elif self.stepNum < self.nStep-1:
            self.step(self.stepNum+1)
        else:
            self.stepNum == self.nStep-1
        self.AutoRun()
        

    def Prev(self):
        if self.stepNum <= 0:
            self.stepNum = 0
        elif self.stepNum == 4:
            for wg in self.StepWidgets[self.stepNum]:
                wg.setVisible(False)
            self.stepNum -= 1
            if self.bandA and (self.subStepNum == 2 or self.subStepNum == 3):
                self.subStepNum = 1
                self.subStep(self.subStepNum)
            else:
                self.subStep(self.subStepNum)
        elif self.stepNum == 3 and self.subStepNum == 0:
            for wg in self.subStepWidgets[self.subStepNum]:
                wg.setVisible(False)
            self.subStepNum = -1
            self.step(self.stepNum-1)
        elif self.stepNum == 3 and self.subStepNum > 0:
            self.subStep(self.subStepNum-1)
        else:
            self.step(self.stepNum-1)
        self.AutoRun()

    def AutoRun(self):
        if self.stepNum == 1:
            if self.initS1:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_s1_run.setEnabled(False)
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.s1_Run()
                self.initS1 = False
                self.B_s1_run.setEnabled(True)
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
        elif self.stepNum == 2:
            if self.initS2:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_s2_run.setEnabled(False)
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.s2_Run()
                self.initS2 = False
                self.B_s2_run.setEnabled(True)
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
        elif self.stepNum == 3 and self.subStepNum == 0:
            if self.initS3_1:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_s3_1_run.setEnabled(False)
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.s3_1_Run()
                self.initS3_1 = False
                self.B_s3_1_run.setEnabled(True)
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
        elif self.stepNum == 3 and self.subStepNum == 1:
            if self.initS3_2:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.B_s3_wvCal.setEnabled(False)
                self.B_s3_wvShow.setEnabled(False)
                self.B_s3_2_FRapply.setEnabled(False)
                self.B_s3_2_simple.setEnabled(False)
                self.B_s3_2_FringeShow.setEnabled(False)
                self.B_s3_2_resShow.setEnabled(False)
                self.B_s3_3_Blink.setEnabled(False)
                self.s3_2_wvCal()
                self.s3_2_simple()
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
                self.B_s3_wvCal.setEnabled(True)
                self.B_s3_wvShow.setEnabled(True)
                self.B_s3_2_FRapply.setEnabled(True)
                self.B_s3_2_simple.setEnabled(True)
                self.B_s3_2_FringeShow.setEnabled(True)
                self.B_s3_2_resShow.setEnabled(True)
                self.B_s3_3_Blink.setEnabled(True)
                self.initS3_2 = False
        elif self.stepNum == 3 and self.subStepNum == 2:
            if self.initS3_3:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_s3_3_run.setEnabled(False)
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.s3_3_Run()
                self.initS3_3 = False
                self.B_s3_3_run.setEnabled(True)
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
        elif self.stepNum == 3 and self.subStepNum == 3:
            if self.initS3_4:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.B_s3_4_wvCal.setEnabled(False)
                self.B_s3_4_wvShow.setEnabled(False)
                self.B_s3_4_FRapply.setEnabled(False)
                self.B_s3_4_simple.setEnabled(False)
                self.B_s3_4_gauss.setEnabled(False)
                self.B_s3_4_reset.setEnabled(False)
                self.B_s3_4_FringeShow.setEnabled(False)
                self.B_s3_4_resShow.setEnabled(False)
                self.B_s3_4_blink.setEnabled(False)
                self.s3_4_wvCal()
                self.s3_4_simple()
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
                self.B_s3_4_wvCal.setEnabled(True)
                self.B_s3_4_wvShow.setEnabled(True)
                self.B_s3_4_FRapply.setEnabled(True)
                self.B_s3_4_simple.setEnabled(True)
                self.B_s3_4_gauss.setEnabled(True)
                self.B_s3_4_reset.setEnabled(True)
                self.B_s3_4_FringeShow.setEnabled(True)
                self.B_s3_4_resShow.setEnabled(True)
                self.B_s3_4_blink.setEnabled(True)
                self.initS3_4 = False
        elif self.stepNum == 4:
            if self.initS4:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_s4_run.setEnabled(False)
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.s4_Run()
                self.initS4 = False
                self.B_s4_run.setEnabled(True)
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)
        elif self.stepNum == 5:
            if self.initS5:
                self.log += "> Running automatically<br>> Please wait.<br>"
                self._writeLog()
                self.B_s5_Run.setEnabled(False)
                self.B_Next.setEnabled(False)
                self.B_Prev.setEnabled(False)
                self.s5_Run()
                self.initS5 = False
                self.B_s5_Run.setEnabled(True)
                self.B_Next.setEnabled(True)
                self.B_Prev.setEnabled(True)



    def s0_step7(self):
        self.step(7)
        qSleep(0.05)
        
    def s0_pf(self):
        if self.frameNum <= 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf-1
        else:
            self.frameNum -= 1
        self.LE_s0_frame.setText(f"{self.frameNum+1}")

    def s0_nf(self):
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf-1:
            self.frameNum = self.CF.nf -1
        else:
            self.frameNum += 1
        self.LE_s0_frame.setText(f"{self.frameNum+1}")

    def _txtCH_LE_s0(self):
        self.frameNum = int(self.LE_s0_frame.text())-1
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf -1
        self.im_s0.set_data(self.CF.logRF[self.frameNum,5:-5,5:-5])
        self.fig.canvas.draw_idle()

    def s1_Run(self):
        self.log += "> Get tilt angle automatically.<br>"
        self._writeLog()
        self.CF.tilt = proc_base.get_tilt(self.s1_data)
        self.s1_img(self.s1_data)
        self.B_s1_Apply.setEnabled(True)

    def s1_img(self, img):
        self.log += f"> Tilt: {self.CF.tilt:.3f} degree.<br>"
        self._writeLog()
        self.LE_s1_tilt.setText(f"{self.CF.tilt:.3f}")
        self.CF.rlRF = proc_base.tilt_correction(self.CF.logRF, self.CF.tilt, cubic=False)
        self.log += f"> Done.<br>"
        self._writeLog()
        # draw results
        dy_img = np.gradient(img, axis=0)
        wp = 40
        whd = np.abs(dy_img[20:-20,wp:wp+20].mean(1)).argmax() + 20
        rimg = proc_base.tilt_correction(img, self.CF.tilt, cubic=True)
        iimg = img - np.median(img, axis=0)
        irimg = rimg - np.median(rimg, axis=0)
        m = iimg[whd-16:whd+16].mean()
        std = iimg[whd-16:whd+16].std()

        self.ax[1][0].cla()
        self.ax[1][1].cla()
        imo = self.ax[1][0].imshow(iimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        imr = self.ax[1][1].imshow(irimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        imo.set_clim(m-std*2, m+std*2)
        imr.set_clim(m-std*2, m+std*2)

        self.ax[1][0].set_aspect(adjustable='box', aspect='auto')
        self.ax[1][1].set_aspect(adjustable='box', aspect='auto')

        self.ax[1][0].set_xlabel('Wavelength (pix)')
        self.ax[1][1].set_xlabel('Wavelength (pix)')
        self.ax[1][0].set_ylabel('Slit (pix)')
        self.ax[1][0].set_title('Original Image')
        self.ax[1][1].set_title('Corrected Image')
        self.ax[1][0].set_ylim(whd-10,whd+10)

        # self.fig.tight_layout(w_pad=0.1)
        self.fig.canvas.draw_idle()

    def s1_Apply(self):
        self.CF.tilt = float(self.LE_s1_tilt.text())
        self.s1_img(self.s1_data)
        
    def s2_Run(self):
        self.log += "> Calculate curvature coefficient automatically.<br>"
        self._writeLog()
        self.CF.coeff, self.dw = proc_base.get_curve_par(self.CF.rlRF)
        self.LE_s2_p0.setText(f"{self.CF.coeff[0]:.3e}")
        self.LE_s2_p1.setText(f"{self.CF.coeff[1]:.3e}")
        self.LE_s2_p2.setText(f"{self.CF.coeff[2]:.3e}")
        self.log += f"> Done.<br>"
        self._writeLog()
        self.s2_make()
        self.B_s1_Apply.setEnabled(True)
    
    def s2_make(self):
        self.CF.logF, oimg, cimg, wh = proc_base.curvature_correction(self.CF.rlRF, self.CF.coeff, show=True)

        y = np.arange(self.CF.rlRF.shape[1])
        wf = np.polyval(self.CF.coeff, y)

        for ax in self.ax[2]:
            ax.cla()

        p1 = f"$+{self.CF.coeff[1]:.2e}x$" if np.sign(self.CF.coeff[1]) == 1 else f"${self.CF.coeff[1]:.2e}x$"
        p2 = f"$+{self.CF.coeff[2]:.2e}" if np.sign(self.CF.coeff[2]) == 1 else f"${self.CF.coeff[2]:.2e}"
        eq = f"$y = {self.CF.coeff[0]:.2e}x^2${p1}{p2}"
        eq = eq.replace('e','^{')
        eq = eq.replace('x','}x')
        eq = eq + '}$'
        self.ax[2][0].scatter(y, self.dw, marker='+')
        self.ax[2][0].plot(y, wf, color='r', label=eq)
        self.ax[2][0].set_xlabel('Slit (pix)')
        self.ax[2][0].set_ylabel('dw (pix)')
        self.ax[2][0].set_title('Curvature')
        self.ax[2][0].legend()

        m = oimg[5:-5,wh-10:wh+10].mean()
        std = oimg[5:-5,wh-10:wh+10].std()
        oim = self.ax[2][1].imshow(oimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        cim = self.ax[2][2].imshow(cimg, plt.cm.gray, origin='lower', interpolation='bilinear')

        oim.set_clim(m-std*1.5, m+std*1.5)
        cim.set_clim(m-std*1.5, m+std*1.5)

        self.ax[2][1].set_xlim(wh-10,wh+10)
        self.ax[2][1].set_aspect(adjustable='box', aspect='auto')
        self.ax[2][2].set_aspect(adjustable='box', aspect='auto')
        self.ax[2][1].set_xlabel('Wavelength (pix)')
        self.ax[2][2].set_xlabel('Wavelength (pix)')
        self.ax[2][1].set_ylabel('Slit (pix)')
        self.ax[2][1].set_title('Original')
        self.ax[2][2].set_title('Corrected')

        self.fig.canvas.draw_idle()

    def s2_Apply(self):
        self.log += "> Apply coefficient.<br>"
        self._writeLog()
        self.CF.coeff[0] = float(self.LE_s2_p0.text())
        self.CF.coeff[1] = float(self.LE_s2_p1.text())
        self.CF.coeff[2] = float(self.LE_s2_p2.text())
        self.s2_make()

    def s3_1_Run(self):
        self.log += "> Run Atlas subtraction.<br>"
        self._writeLog()

        self.CF.atlas_subtraction()
        self.log += f"> Done.<br>"
        self._writeLog()
        for ax in self.ax_sub[0]:
            ax.cla()

        data = self.CF.rmFlat[self.frameNum,5:-5,5:-5]
        m = data.mean()
        std = data.std()
        self.im_s3_1 = self.ax_sub[0][0].imshow(data, plt.cm.gray, origin='lower')
        self.ax_sub[0][0].set_xlabel('Wavelength (pix)')
        self.ax_sub[0][0].set_ylabel('Slit (pix)')
        self.ax_sub[0][0].set_title('Atlas Subtraction')
        self.im_s3_1.set_clim(m-std*1.5, m+std*1.5)
        self.fig.canvas.draw_idle()

    def s3_1_pf(self):
        if self.frameNum <= 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf-1
        else:
            self.frameNum -= 1
        self.LE_s3_1_frame.setText(f"{self.frameNum+1}")

    def s3_1_nf(self):
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf-1:
            self.frameNum = self.CF.nf -1
        else:
            self.frameNum += 1
        self.LE_s3_1_frame.setText(f"{self.frameNum+1}")

    def _txtCH_LE_s3_1(self):
        self.frameNum = int(self.LE_s3_1_frame.text())-1
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf -1
        self.im_s3_1.set_data(self.CF.rmFlat[self.frameNum,5:-5,5:-5])
        self.fig.canvas.draw_idle()

    def s3_2_wvCal(self):
        self.log += "> Wavelet calculation.<br>"
        self._writeLog()
        self.wvlet_y = [None]*self.CF.nf

        for i in range(self.CF.nf):
            self.wvlet_y[i] = Wavelet(self.CF.rmFlat[i], dt=1, axis=0, dj=0.05, param=12)

        self.log += "> Done.<br>"
        self._writeLog()
        self.s3_2_wvShow()
        self.B_s3_wvShow.setEnabled(True)
        self.B_s3_2_FRapply.setEnabled(True)
        self.B_s3_2_simple.setEnabled(True)

    def s3_2_wvShow(self):
        self.ax_sub[1][0].set_visible(True)
        self.ax_sub[1][1].set_visible(False)
        nfreq = self.wvlet_y[self.frameNum].wavelet.shape[1]
        self.ax_sub[1][0].cla()
        data = np.abs(self.wvlet_y[self.frameNum].wavelet).mean((0,2))
        ymax = data[:nfreq-10].max()*1.1
        self.ax_sub[1][0].plot(data)
        self.ax_sub[1][0].set_ylim(0,ymax)
        self.ax_sub[1][0].set_xlim(-0.5, nfreq-1)
        self.ax_sub[1][0].set_xlabel('freq_y (pix)')
        self.ax_sub[1][0].set_ylabel('Amplitude')
        self.ax_sub[1][0].set_title('Averaged Wavelet Spectrum (y-dir)')
        self.pFRmin_y = self.ax_sub[1][0].plot([self.yf_min, self.yf_min], [0, ymax], color='r', ls='dashed')[0]
        self.pFRmax_y = self.ax_sub[1][0].plot([self.yf_max, self.yf_max], [0, ymax], color='r', ls='dashed')[0]
        self.fig.canvas.draw_idle()

    def s3_2_FRapply(self):
        self.yf_min = int(self.LE_s3_2_FRmin.text())
        self.yf_max = int(self.LE_s3_2_FRmax.text())
        self.pFRmin_y.set_xdata([self.yf_min, self.yf_min])
        self.pFRmax_y.set_xdata([self.yf_max, self.yf_max])
        self.log += "> Change Frequency Range.<br>"
        self.log += "> Please press the calculate button again to apply this to Fringe pattern.<br>"
        self._writeLog()

    def s3_2_simple(self):
        self.YFshow = True
        self.log += "> Calculate Fringe Patterns.<br>"
        self._writeLog()
        self.yFringe = np.zeros([self.CF.nf,self.CF.ny,self.CF.nw])
        self.s1 = np.zeros([self.CF.nf,self.CF.ny,self.CF.nw])
        self.ms1 = np.zeros([self.CF.nf,self.CF.ny,self.CF.nw])
        for i in range(self.CF.nf):
            self.yFringe[i] = proc_base.cal_fringeSimple(self.wvlet_y[i], [self.yf_min, self.yf_max]).T
            
        # self.yFringe = proc_base.YFart_correction(self.yFringe)
        self.yFringe -= self.yFringe[:,5:-5,5:-5].mean((1,2))[:,None,None]

        self.s1 = self.CF.rmFlat - self.yFringe
        self.ms1 = self.s1.copy()
        self.log += "> Done.<br>"
        self._writeLog()
        if self.imFy is None:
            self.imFy = self.ax_sub[1][1].imshow(self.yFringe[self.frameNum], plt.cm.gray, origin='lower')
            self.ax_sub[1][1].set_title('Fringe Pattern')
            self.ax_sub[1][1].set_xlabel('Wavelength (pix)')
            self.ax_sub[1][1].set_ylabel('Slit (pix)')
        else:
            self.ax_sub[1][1].set_title('Fringe Pattern')
            self.imFy.set_data(self.yFringe[self.frameNum][5:-5,5:-5])
            self.imFy.set_clim(self.yFringe[self.frameNum][5:-5,5:-5].min(),self.yFringe[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()
        self.B_s3_2_FringeShow.setEnabled(True)
        self.B_s3_2_resShow.setEnabled(True)
        self.B_s3_2_blink.setEnabled(True)

    def s3_2_FringeShow(self):
        self.YFshow = True
        self.ax_sub[1][0].set_visible(False)
        self.ax_sub[1][1].set_visible(True)
        self.ax_sub[1][1].set_title('Fringe Pattern (y-dir)')
        self.imFy.set_data(self.yFringe[self.frameNum][5:-5,5:-5])
        self.imFy.set_clim(self.yFringe[self.frameNum][5:-5,5:-5].min(),self.yFringe[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()

    def s3_2_resShow(self):
        self.YFshow = False
        self.s1Show = True        
        self.ax_sub[1][0].set_visible(False)
        self.ax_sub[1][1].set_visible(True)
        self.ax_sub[1][1].set_title('y-dir Fringe Subtracted Image')
        self.imFy.set_data(self.s1[self.frameNum][5:-5,5:-5])
        self.imFy.set_clim(self.s1[self.frameNum][5:-5,5:-5].min(),self.s1[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()

    def s3_2_blink(self):
        self.s1Show = not self.s1Show
        if self.YFshow:
            self.YFshow = False
            self.imFy.set_clim(self.s1[self.frameNum][5:-5,5:-5].min(),self.s1[self.frameNum][5:-5,5:-5].max())
        if self.s1Show:
            self.ax_sub[1][1].set_title('y-dir Fringe Subtracted Image')
            self.imFy.set_data(self.s1[self.frameNum][5:-5,5:-5])
        else:
            self.ax_sub[1][1].set_title('Original Image')
            self.imFy.set_data(self.CF.rmFlat[self.frameNum][5:-5,5:-5])
        
        self.fig.canvas.draw_idle()
        
    def s3_2_pf(self):
        if self.frameNum <= 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf-1
        else:
            self.frameNum -= 1
        self.LE_s3_2_frame.setText(f"{self.frameNum+1}")

    def s3_2_nf(self):
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf-1:
            self.frameNum = self.CF.nf -1
        else:
            self.frameNum += 1
        self.LE_s3_2_frame.setText(f"{self.frameNum+1}")

    def _txtCH_LE_s3_2(self):
        self.frameNum = int(self.LE_s3_2_frame.text())-1
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf -1
        if self.YFshow:
            data = self.yFringe[self.frameNum][5:-5,5:-5]
        elif self.s1Show:
            data = self.s1[self.frameNum][5:-5,5:-5]
        else:
            data = self.CF.rmFlat[self.frameNum][5:-5,5:-5]

        self.imFy.set_data(data)
        self.fig.canvas.draw_idle()

    def s3_3_Run(self):
        self.log += f"> Automatically find mask center and width.<br>"
        self._writeLog()
        self.cpos = self.CF.mlf[:,5:-5].argmin(1) + 5
        w = np.zeros(self.CF.nf, dtype=int)
        x = np.arange(self.CF.nw)
        for i in range(self.CF.nf):
            # tt = self.s1[i][5:-5,5:-5].mean(0) - self.s1[i][5:-5,5:-5].min()
            tt = self.CF.mlf[i,5:-5] - self.CF.mlf[i,5:-5].max()
            whmin = self.cpos[i]
            pars = [tt[whmin-5], whmin, 5]
            cp, cr = curve_fit(proc_base.Gaussian, x[whmin-5:whmin+5], tt[whmin-5-5:whmin-5+5], p0=pars)
            # self.cpos[i] = int(cp[1])
            w[i] = int(cp[2]*1.2)

        self.msk_width = int(np.median(w))
        self.ms1 = self.s1.copy()

        lF = self.CF.logF - self.yFringe
        msk = proc_base.getMask(10**lF, power=6, fsig=1)
        msk[...,-5:]=1
        msk[...,:5]=1
        msk = msk[...,5:-5]
        lF = lF[...,5:-5]
        s1 = self.s1[...,5:-5].copy()
        wh = msk >= 0.3
        bl = msk < 0.3
        x = np.arange(s1.shape[-1])
        lag = 10
        self.log += f"> Loading 0/{self.CF.nf} frame<br>"
        self.log += f"    >> 0%<br>"
        for i in range(self.CF.nf):
            self.log = self.log.replace(f"{i}/{self.CF.nf}", f"{i+1}/{self.CF.nf}")
            # ARcast (forecasting method)
            x1 = x[bl[i,120]]
            x2 = np.roll(x[bl[i,120]],1)
            kk = np.arange(len(x1))[x1-x2 != 1]
            nsp = len(kk)
            for j in range(nsp):
                self.log = self.log.replace(f">> 100%", f">> 0%")
                self.log = self.log.replace(f">> {j/nsp*100:.0f}%", f">> {(j+1)/nsp*100:.0f}%")
                self._writeLog()
                if j == nsp-1:
                    npredict = x1[-1] - x1[kk[j]] +1
                else:
                    npredict = kk[j+1] - kk[j]
                ii = x1[kk[j]]//2-1
                flag = lag if lag < ii else int(ii)
                fskip = False
                if int(ii) <= lag*1.5:
                    fskip = True
                elif int(ii) < lag*2:
                    flag = 5
                
                if j != nsp-1:
                    bst = x1[kk[j+1]-1]+1
                else:
                    bst = x1[-1]+1
                ii = (s1.shape[-1]-bst)//2-1
                blag = lag if lag < ii else int(ii)
                bskip = False
                if int(ii) <= lag*1.5:
                    bskip = True
                elif int(ii) < lag*2:
                    blag = 5
                fp = np.zeros(npredict)
                bp = np.zeros(npredict)
                bF0 = (np.arange(npredict)+1)/(npredict+1)
                fF0 = 1 - bF0
                bF = (1-bskip)*(bF0 + fskip*fF0)
                fF = (1-fskip)*(fF0 + bskip*bF0)
                
                for l in range(s1.shape[1]):
                    # foreward
                    if not fskip:
                        fm = AutoReg(s1[i,l,:x1[kk[j]]],lags=5).fit()
                        fp = fm.forecast(npredict)
                    # backward
                    if not bskip:
                        bm = AutoReg(s1[i,l,bst:][::-1],lags=5).fit()
                        bp = bm.forecast(npredict)
                    pred = fF*fp + bF*bp[::-1]
                    if j != nsp-1:
                        self.ms1[i,l,x1[kk[j]]+5:x1[kk[j+1]-1]+1+5] = pred
                        # s1[i,l,x1[kk[j]]:x1[kk[j+1]-1]+1] = pred
                    else:
                        self.ms1[i,l,x1[kk[j]]+5:x1[-1]+1+5] = pred
                        # s1[i,l,x1[kk[j]]:x1[-1]+1] = pred
                    
            # ARcast prediction
            # x1 = x[bl[i,120]]
            # x2 = np.roll(x[bl[i,120]],1)
            # kk = np.arange(len(x1))[x1-x2 != 1]
            # nsp = len(kk)
            # for j in range(nsp):
            #     self.log = self.log.replace(f">> 100%", f">> 0%")
            #     self.log = self.log.replace(f">> {j/nsp*100:.0f}%", f">> {(j+1)/nsp*100:.0f}%")
            #     self._writeLog()
                
            #     for l in range(s1.shape[1]):
            #         tm = AutoReg(s1[i,l], lags=lag).fit()
            #         if j != nsp-1:
            #             pred = tm.predict(x1[kk[j]],x1[kk[j+1]-1])
            #             self.ms1[i,l,x1[kk[j]]+5:x1[kk[j+1]-1]+1+5] = pred
            #         else:
            #             pred = tm.predict(x1[kk[j]],x1[-1])
            #             self.ms1[i,l,x1[kk[j]]+5:x1[-1]+1+5] = pred
            
            # # interp
            # inp = interp1d(x[wh[i,120]], self.s1[...,wh[i,120]][i], axis=1, kind='nearest', fill_value='extrapolate')
            # self.ms1[i] = inp(x)
            ## ori
            # mskMin = self.cpos[i]-self.msk_width
            # mskMin = mskMin if mskMin >= 0 else 0
            # mskMax = self.cpos[i]+self.msk_width
            # mskMax = mskMax if mskMax <= self.CF.nw-1 else self.CF.nw-1
            # self.ms1[i] = proc_base.data_mask_and_fill(self.s1[i], [[mskMin], [mskMax]])

        self.mskShow = True
        self.log += f"> Done.<br>"
        self._writeLog()
        if self.im_s3_3 is None:
            self.im_s3_3 = self.ax_sub[2][0].imshow(self.ms1[self.frameNum][5:-5,5:-5], plt.cm.gray, origin='lower')
            
            self.p_s3_3_mskMin = self.ax_sub[2][0].plot([self.cpos[self.frameNum]-5-self.msk_width, self.cpos[self.frameNum]-5-self.msk_width], [0,self.CF.ny-11], color='r', ls='dashed')[0]
            self.p_s3_3_mskMax = self.ax_sub[2][0].plot([self.cpos[self.frameNum]-5+self.msk_width, self.cpos[self.frameNum]-5+self.msk_width], [0,self.CF.ny-11], color='r', ls='dashed')[0]
            self.ax_sub[2][0].set_xlabel('Wavelength (pix)')
            self.ax_sub[2][0].set_ylabel('Slit (pix)')
            self.ax_sub[2][0].set_title('Masked Image')
        else:
            self.im_s3_3.set_data(self.ms1[self.frameNum][5:-5,5:-5])
            self.p_s3_3_mskMin.set_xdata([self.cpos[self.frameNum]-5-self.msk_width, self.cpos[self.frameNum]-5-self.msk_width])
            self.p_s3_3_mskMax.set_xdata([self.cpos[self.frameNum]-5+self.msk_width, self.cpos[self.frameNum]-5+self.msk_width])
        cm = self.s1[...,5:-5].mean()
        cstd = self.s1[...,5:-5].std()
        self.im_s3_3.set_clim(cm-cstd*2, cm+cstd*2)
        self.B_s3_3_Blink.setEnabled(True)
        self.fig.canvas.draw_idle()



    def s3_3_Blink(self):
        self.mskShow = not self.mskShow

        if self.mskShow == True:
            self.im_s3_3.set_data(self.ms1[self.frameNum][5:-5,5:-5])
            self.ax_sub[2][0].set_title('Masked Image')
        else:
            self.im_s3_3.set_data(self.s1[self.frameNum][5:-5,5:-5])
            self.ax_sub[2][0].set_title('Original Image')

        self.fig.canvas.draw_idle()

    def s3_3_reset(self):
        self.log += f"> Remove the mask.<br>"
        self._writeLog()
        self.msk_wdith = 0
        self.ms1 = self.s1.copy()

        if self.mskShow == True:
            self.im_s3_3.set_data(self.ms1[self.frameNum][5:-5,5:-5])
            self.ax_sub[2][0].set_title('Masked Image')
        else:
            self.im_s3_3.set_data(self.s1[self.frameNum][5:-5,5:-5])
            self.ax_sub[2][0].set_title('Original Image')

        self.fig.canvas.draw_idle()

    def s3_3_pf(self):
        if self.frameNum <= 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf-1
        else:
            self.frameNum -= 1
        self.LE_s3_3_frame.setText(f"{self.frameNum+1}")

    def s3_3_nf(self):
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf-1:
            self.frameNum = self.CF.nf -1
        else:
            self.frameNum += 1
        self.LE_s3_3_frame.setText(f"{self.frameNum+1}")

    def _txtCH_LE_s3_3(self):
        self.frameNum = int(self.LE_s3_3_frame.text())-1
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf -1
        if self.mskShow == True:
            self.im_s3_3.set_data(self.ms1[self.frameNum][5:-5,5:-5])
        else:
            self.im_s3_3.set_data(self.s1[self.frameNum][5:-5,5:-5])
        self.p_s3_3_mskMin.set_xdata([self.cpos[self.frameNum]-5-self.msk_width, self.cpos[self.frameNum]-5-self.msk_width])
        self.p_s3_3_mskMax.set_xdata([self.cpos[self.frameNum]-5+self.msk_width, self.cpos[self.frameNum]-5+self.msk_width])

        self.fig.canvas.draw_idle()

    def s3_4_wvCal(self):
        self.log += "> Wavelet calculation.<br>"
        self._writeLog()
        self.wvlet_x = [None]*self.CF.nf
        w = np.zeros(self.CF.nf)
        xwh = np.zeros(self.CF.nf, dtype=int)
        x = np.arange(14)
        for i in range(self.CF.nf):
            self.wvlet_x[i] = Wavelet(self.ms1[i][5:-5,5:-5], dt=1, axis=1, dj=0.05, param=12)
            data = np.abs(self.wvlet_x[i].wavelet).mean((0,2))
            xwh[i] = data[:-25].argmax()
            hmax = data[:-25].max()/2
            t = x[data[xwh[i]-7:xwh[i]+7] >= hmax]
            w[i] = (t.max() - t.min())/2

        self.xF_hw = int(w.mean()*3)
        self.xF_c = int(xwh.mean())

            


        self.log += "> Done.<br>"
        self._writeLog()
        self.s3_4_wvShow()
        self.B_s3_4_wvShow.setEnabled(True)
        self.B_s3_4_FRapply.setEnabled(True)
        self.B_s3_4_simple.setEnabled(True)

    def s3_4_wvShow(self):
        self.xwvShow = True
        self.XFshow = False
        self.s2Show = False
        self.ax_sub[3][0].set_visible(True)
        self.ax_sub[3][1].set_visible(False)
        nfreq = self.wvlet_x[self.frameNum].wavelet.shape[1]
        self.ax_sub[3][0].cla()
        data = np.abs(self.wvlet_x[self.frameNum].wavelet).mean((0,2))
        # xwh = data[:nfreq-20].argmax()
        self.xf_min = self.xF_c - self.xF_hw
        self.xf_max = self.xF_c + self.xF_hw
        self.LE_s3_4_FRmin.setText(f"{self.xf_min}")
        self.LE_s3_4_FRmax.setText(f"{self.xf_max}")
        ymax = data[:nfreq-10].max()*1.1
        self.pwvx = self.ax_sub[3][0].plot(data)[0]
        self.ax_sub[3][0].set_ylim(0,ymax)
        self.ax_sub[3][0].set_xlim(-0.5, nfreq-1)
        self.ax_sub[3][0].set_xlabel('Freq_x (pix)')
        self.ax_sub[3][0].set_ylabel('Amplitude')
        self.ax_sub[3][0].set_title('Averaged Wavelet Spectrum (x-dir)')
        self.pFRmin_x = self.ax_sub[3][0].plot([self.xf_min, self.xf_min], [0, ymax], color='r', ls='dashed')[0]
        self.pFRmax_x = self.ax_sub[3][0].plot([self.xf_max, self.xf_max], [0, ymax], color='r', ls='dashed')[0]
        self.fig.canvas.draw_idle()

    def s3_4_FRapply(self):
        self.xf_min = int(self.LE_s3_4_FRmin.text())
        self.xf_max = int(self.LE_s3_4_FRmax.text())
        self.pFRmin_x.set_xdata([self.xf_min, self.xf_min])
        self.pFRmax_x.set_xdata([self.xf_max, self.xf_max])
        self.log += "> Change Frequency Range.<br>"
        self.log += "> Please press the calculate button again to apply this to Fringe pattern.<br>"
        self._writeLog()

    def s3_4_simple(self):
        self.XFshow = True
        self.xwvShow = False
        self.log += "> Calculate Fringe Patterns in simple way.<br>"
        self._writeLog()
        self.xFringe = np.zeros([self.CF.nf,self.CF.ny,self.CF.nw])
        self.s2 = np.zeros((self.CF.nf, self.CF.ny, self.CF.nw))
        for i in range(self.CF.nf):
            self.xFringe[i][5:-5,5:-5] = proc_base.cal_fringeSimple(self.wvlet_x[i], [self.xf_min, self.xf_max])
            self.xFringe[i][5:-5,5:-5] -= self.xFringe[i][5:-5,5:-5].mean()
            self.s2[i][5:-5,5:-5] = self.s1[i][5:-5,5:-5] - self.xFringe[i][5:-5,5:-5]
        self.log += "> Done.<br>"
        self._writeLog()
        if self.imFx is None:
            self.imFx = self.ax_sub[3][1].imshow(self.xFringe[self.frameNum][5:-5,5:-5], plt.cm.gray, origin='lower')
            self.ax_sub[3][1].set_title('Fringe Pattern (x-dir)')
            self.ax_sub[3][1].set_xlabel('Wavelength (pix)')
            self.ax_sub[3][1].set_ylabel('Slit (pix)')
        else:
            self.ax_sub[3][1].set_title('Fringe Pattern (x-dir)')
            self.imFx.set_data(self.xFringe[self.frameNum][5:-5,5:-5])
            self.imFx.set_clim(self.xFringe[self.frameNum][5:-5,5:-5].min(),self.xFringe[self.frameNum].max())
        self.fig.canvas.draw_idle()
        self.B_s3_4_FringeShow.setEnabled(True)
        self.B_s3_4_resShow.setEnabled(True)
        self.B_s3_4_blink.setEnabled(True)

    def s3_4_gauss(self):
        self.XFshow = True
        self.log += "> Calculate Fringe Patterns by using gaussian fit.<br>"
        self._writeLog()
        self.xFringe = [None]*self.CF.nf
        self.s2 = np.zeros((self.CF.nf, self.CF.ny, self.CF.nw))
        for i in range(self.CF.nf):
            res = proc_base.cal_fringeGauss(self.wvlet_x[i], [self.xf_min, self.xf_max])
            if res is None:
                self.log += f"> <font color='{self.font_err}'> Cannot Calculate Fringe Patterns by using gaussian fit.</font><br>"
                self._writeLog()
                return 0
            else:
                self.xFringe[i] = res
                self.log += f"> {i}-Frame Done.<br>"
                self._writeLog()
            self.s2[i][5:-5,5:-5] = self.s1[i][5:-5,5:-5] - self.xFringe[i]
        self.log += "> Done.<br>"
        self._writeLog()
        if self.imFx is None:
            self.imFx = self.ax_sub[3][1].imshow(self.xFringe[self.frameNum], plt.cm.gray, origin='lower')
            self.ax_sub[3][1].set_title('Fringe Pattern')
            self.ax_sub[3][1].set_xlabel('Wavelength (pix)')
            self.ax_sub[3][1].set_ylabel('Slit (pix)')
        else:
            self.ax_sub[3][1].set_title('Fringe Pattern')
            self.imFx.set_data(self.xFringe[self.frameNum])
            self.imFx.set_clim(self.xFringe[self.frameNum].min(),self.xFringe[self.frameNum].max())
        self.fig.canvas.draw_idle()
        self.B_s3_4_FringeShow.setEnabled(True)
        self.B_s3_4_resShow.setEnabled(True)
        self.B_s3_4_blink.setEnabled(True)

    def s3_4_FringeShow(self):
        self.XFshow = True
        self.xwvShow = False
        self.ax_sub[3][0].set_visible(False)
        self.ax_sub[3][1].set_visible(True)
        self.ax_sub[3][1].set_title('Fringe Pattern (x-dir)')
        self.imFx.set_data(self.xFringe[self.frameNum][5:-5,5:-5])
        self.imFx.set_clim(self.xFringe[self.frameNum][5:-5,5:-5].min(),self.xFringe[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()

    def s3_4_resShow(self):
        self.XFshow = False
        self.s2Show = True 
        self.xwvShow = False      
        self.ax_sub[3][0].set_visible(False)
        self.ax_sub[3][1].set_visible(True)
        self.ax_sub[3][1].set_title('x-dir Fringe Subtracted Image')
        self.imFx.set_data(self.s2[self.frameNum][5:-5,5:-5])
        self.imFx.set_clim(self.s2[self.frameNum][5:-5,5:-5].min(),self.s2[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()

    def s3_4_blink(self):
        self.s2Show = not self.s2Show
        self.xwvShow = False
        if self.XFshow:
            self.XFshow = False
            self.imFx.set_clim(self.s2[self.frameNum][5:-5,5:-5].min(),self.s2[self.frameNum][5:-5,5:-5].max())
        if self.s2Show:
            self.ax_sub[3][1].set_title('x-dir Fringe Subtracted Image')
            self.imFx.set_data(self.s2[self.frameNum][5:-5,5:-5])
        else:
            self.ax_sub[3][1].set_title('Original Image')
            self.imFx.set_data(self.CF.rmFlat[self.frameNum][5:-5,5:-5])
        
        self.fig.canvas.draw_idle()

    def s3_4_reset(self):
        self.log += f"> Reset.<br>"
        self._writeLog()
        
        
        self.ms1 = self.s1.copy()
        self.s2 = None
        self.xFringe = None

        self.ax_sub[3][0].cla()
        self.ax_sub[3][1].cla()

        self.fig.canvas.draw_idle()


    def s3_4_pf(self):
        if self.frameNum <= 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf-1
        else:
            self.frameNum -= 1
        self.LE_s3_4_frame.setText(f"{self.frameNum+1}")

    def s3_4_nf(self):
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf-1:
            self.frameNum = self.CF.nf -1
        else:
            self.frameNum += 1
        self.LE_s3_4_frame.setText(f"{self.frameNum+1}")

    def _txtCH_LE_s3_4(self):
        self.frameNum = int(self.LE_s3_4_frame.text())-1
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf -1
        if self.XFshow:
            data = self.xFringe[self.frameNum][5:-5,5:-5]
            self.imFx.set_data(data)
        elif self.s2Show:
            data = self.s2[self.frameNum][5:-5,5:-5]
            self.imFx.set_data(data)
        elif self.xwvShow:
            self.ax_sub[3][0].set_visible(True)
            self.ax_sub[3][1].set_visible(False)
            data = np.abs(self.wvlet_x[self.frameNum].wavelet).mean((0,2))
            self.pwvx.set_ydata(data)
        else:
            data = self.CF.rmFlat[self.frameNum][5:-5,5:-5]
            self.imFx.set_data(data)

        self.fig.canvas.draw_idle()

    def s4_Run(self):
        self.log += "> Calculate curvature coefficient automatically.<br>"
        if self.s2 is None:
            self.FS_logF = self.CF.logF - self.yFringe
        else:
            self.FS_logF = self.CF.logF - self.yFringe - self.xFringe
        self._writeLog()
        self.CF.coeff2, self.dw2 = proc_base.get_curve_par(self.FS_logF)
        self.LE_s4_p0.setText(f"{self.CF.coeff2[0]:.3e}")
        self.LE_s4_p1.setText(f"{self.CF.coeff2[1]:.3e}")
        self.LE_s4_p2.setText(f"{self.CF.coeff2[2]:.3e}")
        self.s4_make()
        self.B_s4_Apply.setEnabled(True)
    
    def s4_make(self):
        
        self.CF.logF2, oimg, cimg, wh = proc_base.curvature_correction(self.FS_logF, self.CF.coeff2, show=True)

        y = np.arange(self.CF.rlRF.shape[1])
        wf = np.polyval(self.CF.coeff2, y)

        for ax in self.ax[4]:
            ax.cla()

        p1 = f"$+{self.CF.coeff2[1]:.2e}x$" if np.sign(self.CF.coeff2[1]) == 1 else f"${self.CF.coeff2[1]:.2e}x$"
        p2 = f"$+{self.CF.coeff2[2]:.2e}" if np.sign(self.CF.coeff2[2]) == 1 else f"${self.CF.coeff2[2]:.2e}"
        eq = f"$y = {self.CF.coeff2[0]:.2e}x^2${p1}{p2}"
        eq = eq.replace('e','^{')
        eq = eq.replace('x','}x')
        eq = eq + '}$'
        self.ax[4][0].scatter(y, self.dw2, marker='+')
        self.ax[4][0].plot(y, wf, color='r', label=eq)
        self.ax[4][0].set_xlabel('Slit (pix)')
        self.ax[4][0].set_ylabel('dw (pix)')
        self.ax[4][0].set_title('Curvature')
        self.ax[4][0].legend()

        m = oimg[5:-5,wh-10:wh+10].mean()
        std = oimg[5:-5,wh-10:wh+10].std()
        oim = self.ax[4][1].imshow(oimg, plt.cm.gray, origin='lower', interpolation='bilinear')
        cim = self.ax[4][2].imshow(cimg, plt.cm.gray, origin='lower', interpolation='bilinear')

        oim.set_clim(m-std*1.5, m+std*1.5)
        cim.set_clim(m-std*1.5, m+std*1.5)

        self.ax[4][1].set_xlim(wh-10,wh+10)
        self.ax[4][1].set_aspect(adjustable='box', aspect='auto')
        self.ax[4][2].set_aspect(adjustable='box', aspect='auto')
        self.ax[4][1].set_xlabel('Wavelength (pix)')
        self.ax[4][2].set_xlabel('Wavelength (pix)')
        self.ax[4][1].set_ylabel('Slit (pix)')
        self.ax[4][1].set_title('Original')
        self.ax[4][2].set_title('Corrected')

        self.fig.canvas.draw_idle()

    def s4_Apply(self):
        self.log += "> Apply coefficient.<br>"
        self._writeLog()
        self.CF.coeff2[0] = float(self.LE_s4_p0.text())
        self.CF.coeff2[1] = float(self.LE_s4_p1.text())
        self.CF.coeff2[2] = float(self.LE_s4_p2.text())
        self.s2_make()

    def s5_Run(self):
        self.log += "> Make Flat running.<br>"
        self._writeLog()
        if self.s2 is None:
            self.CF.Flat = self.CF.gain_calib(self.s1)
        else: 
            self.CF.Flat = self.CF.gain_calib(self.s2)  
        # self.CF.Flat = self.CF.gain_calib(self.CF.logF2)
        self.s5_img()

        self.clogF = self.CF.logF2 - np.log10(self.CF.Flat)
        self.ms_cF = 10**(self.clogF - self.clogF[:,5:-5].mean(1)[:,None,:])

        self.B_s5_Show.setEnabled(True)
        self.B_s5_cFlat.setEnabled(True)
        self.B_s5_Blink.setEnabled(True)
        self.B_s5_msFlat.setEnabled(True)
        self.log += "> Done.<br>"
        self._writeLog()

    def s5_img(self):
        # draw results
        if self.im_s5 is None:
            self.im_s5 = self.ax[5][0].imshow(self.CF.Flat[5:-5,5:-5], plt.cm.gray, origin='lower', interpolation='bilinear')
        else:
            self.im_s5.set_data(self.CF.Flat[5:-5,5:-5])

        self.im_s5.set_visible(True)
        self.ax[5][0].set_xlabel('Wavelength (pix)')
        self.ax[5][0].set_ylabel('Slit (pix)')
        self.ax[5][0].set_title('Flat Image')
        self.ax[5][0].set_ylim(-0.5, self.CF.ny-10.5)
        self.ax[5][0].set_aspect(aspect='auto')
        
        if self.p_s5_ori is not None:
            self.p_s5_ori.set_visible(False)
            self.p_s5_ff.set_visible(False)
            self.s5_legend.set_visible(False)

        # self.fig.tight_layout(w_pad=0.1)
        self.fig.canvas.draw_idle()

    def s5_Show(self):
        self.showFlat = True
        self.showCFlat = False
        self.showMSFlat = False
        self.showProf = False

        self.im_s5.set_visible(True)
        self.im_s5.set_data(self.CF.Flat[5:-5,5:-5])
        self.im_s5.set_clim(self.CF.Flat[5:-5,5:-5].min(), self.CF.Flat[5:-5,5:-5].max())
        self.ax[5][0].set_title('Flat Image')
        self.ax[5][0].set_ylabel('Slit (pix)')
        self.ax[5][0].set_ylim(-0.5, self.CF.ny-10.5)
        if self.p_s5_ori is not None:
            self.p_s5_ori.set_visible(False)
            self.p_s5_ff.set_visible(False)
            self.s5_legend.set_visible(False)
        self.fig.canvas.draw_idle()

    def s5_cFlat(self):
        self.showFlat = False
        self.showCFlat = True
        self.showMSFlat = False
        self.showProf = False

        self.im_s5.set_visible(True)
        self.im_s5.set_data(self.clogF[self.frameNum,5:-5,5:-5])
        self.im_s5.set_clim(self.clogF[self.frameNum,5:-5,5:-5].min(), self.clogF[self.frameNum,5:-5,5:-5].max())
        self.ax[5][0].set_title('Flat-fielded Image')
        self.ax[5][0].set_ylabel('Slit (pix)')
        self.ax[5][0].set_ylim(-0.5, self.CF.ny-10.5)
        if self.p_s5_ori is not None:
            self.p_s5_ori.set_visible(False)
            self.p_s5_ff.set_visible(False)
            self.s5_legend.set_visible(False)
        self.fig.canvas.draw_idle()

    def s5_Blink(self):
        self.showCFlat = not self.showCFlat
        self.im_s5.set_visible(True)
        if not self.showFlat and not self.showMSFlat:
            if self.showCFlat:
                self.im_s5.set_data(self.clogF[self.frameNum,5:-5,5:-5])
                self.ax[5][0].set_title('Flat-fielded Image')
            else:
                self.im_s5.set_data(self.CF.logF[self.frameNum,5:-5,5:-5])
                self.ax[5][0].set_title('Original Image')
        self.ax[5][0].set_ylabel('Slit (pix)')
        self.fig.canvas.draw_idle()

    def s5_msShow(self):
        self.showFlat = False
        self.showCFlat = False
        self.showMSFlat = True
        self.showProf = False
        self.im_s5.set_visible(True)
        self.im_s5.set_data(self.ms_cF[self.frameNum,5:-5,5:-5])
        m = self.ms_cF[self.frameNum,5:-5,5:-5].mean()
        std = self.ms_cF[self.frameNum,5:-5,5:-5].std()
        self.im_s5.set_clim(m-std*2, m+std*2)
        self.ax[5][0].set_title('Flat-fielded Image')
        self.ax[5][0].set_ylabel('Slit (pix)')
        self.ax[5][0].set_ylim(-0.5, self.CF.ny-10.5)
        if self.p_s5_ori is not None:
            self.p_s5_ori.set_visible(False)
            self.p_s5_ff.set_visible(False)
            self.s5_legend.set_visible(False)
        self.fig.canvas.draw_idle()

    def s5_profShow(self):
        self.showFlat = False
        self.showCFlat = False
        self.showMSFlat = False
        self.showProf = True

        self.im_s5.set_visible(False)
        ori = self.CF.logF[self.frameNum, 5:-5, 5:-5].mean(0)
        ff = self.clogF[self.frameNum, 5:-5, 5:-5].mean(0)
        if self.p_s5_ori is None:
            self.p_s5_ori = self.ax[5][0].plot(ori, label='original')[0]
            self.p_s5_ff = self.ax[5][0].plot(ff, label='flat-fielded')[0]
            self.s5_legend = self.ax[5][0].legend()
        else:
            self.p_s5_ori.set_ydata(ori)
            self.p_s5_ff.set_ydata(ff)
            self.s5_legend.set_visible(True)
            self.p_s5_ori.set_visible(True)
            self.p_s5_ff.set_visible(True)
        self.ax[5][0].set_title('Profile')
        self.ax[5][0].set_ylabel('Intensity (DN)')
        self.ax[5][0].set_ylim(ff.min()*0.98, ff.max()*1.02)
        self.fig.canvas.draw_idle()


    def s5_pf(self):
        if self.frameNum <= 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf-1
        else:
            self.frameNum -= 1
        self.LE_s5_frame.setText(f"{self.frameNum+1}")

    def s5_nf(self):
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf-1:
            self.frameNum = self.CF.nf -1
        else:
            self.frameNum += 1
        self.LE_s5_frame.setText(f"{self.frameNum+1}")

    def _txtCH_LE_s5(self):
        self.frameNum = int(self.LE_s5_frame.text())-1
        if self.frameNum < 0:
            self.frameNum = 0
        elif self.frameNum >= self.CF.nf:
            self.frameNum = self.CF.nf -1

        if self.showProf:
            ori = self.CF.logF[self.frameNum, 5:-5, 5:-5].mean(0)
            ff = self.clogF[self.frameNum, 5:-5, 5:-5].mean(0)
            self.p_s5_ori.set_ydata(ori)
            self.p_s5_ff.set_ydata(ff)
        elif self.showMSFlat:
            data = self.ms_cF[self.frameNum,5:-5,5:-5]
            self.im_s5.set_data(data)
            self.ax[5][0].set_ylabel('Slit (pix)')
        elif self.showCFlat:
            data = self.clogF[self.frameNum,5:-5,5:-5]
            self.im_s5.set_data(data)
            self.ax[5][0].set_ylabel('Slit (pix)')
        else:
            data = self.CF.logF[self.frameNum,5:-5,5:-5]
            self.im_s5.set_data(data)
            self.ax[5][0].set_ylabel('Slit (pix)')
        self.fig.canvas.draw_idle()

    def s6_save(self):
        ofname = self.fflatGBL[self.fidx]
        flatName = ofname.replace('FISS','FISS_FLAT').replace('_Flat','')
        xFringeName = ofname.replace('FISS','FISS_xFringe').replace('_Flat','')
        yFringeName = ofname.replace('FISS','FISS_yFringe').replace('_Flat','')

        if not isdir(self.pcaldir):
            makedirs(self.pcaldir)

        # reference wavelength value
        self.h = self.CF.h
        wvpar = proc_base.wv_calib_atlas(self.clogF[self.CF.nf//2], self.h)
        self.log += f"> crpix1: {wvpar[0]:.2f}<br> cdelt1: {wvpar[1]:.3f}<br>, crval1: {wvpar[2]:.3f}<br>"
        self._writeLog()
        # header preset
        
        if self.h['STRTIME'].find('.') < 10:
            self.h['STRTIME'] = self.h['STRTIME'].replace('-', 'T').replace('.', '-')
        if self.h['ENDTIME'].find('.') < 10:
            self.h['ENDTIME'] = self.h['ENDTIME'].replace('-', 'T').replace('.', '-')

        obstime = (Time(self.h['STRTIME']).jd + Time(self.h['ENDTIME']).jd)/2
        obstime = Time(obstime, format='jd').isot

        # save Flat
        self.log += "> Save Flat Image.<br>"
        self._writeLog()
        flatHDU = fits.PrimaryHDU(self.CF.Flat)
        flatHDU.header['CRPIX1'] = (wvpar[0], 'reference pixel position')
        flatHDU.header['CDELT1'] = (wvpar[1], 'angstrom/pixel')
        flatHDU.header['CRVAL1'] = (wvpar[2], 'reference wavelength (angstrom)')
        flatHDU.header['EXPTIME'] = (self.h['EXPTIME'], 'Second')
        flatHDU.header['OBSTIME'] = (obstime, 'Observation Time (UT)')
        flatHDU.header['DATE'] = (self.h['DATE'], 'File Creation Date (UT)')
        flatHDU.header['STRTIME'] = (self.h['STRTIME'], 'Scan Start Time')
        flatHDU.header['ENDTIME'] = (self.h['ENDTIME'], 'Scan Finish Time')
        flatHDU.header['TILT'] = (self.CF.tilt, 'Degree')
        flatHDU.header['COEF1_0'] = (self.CF.coeff[0], 'Curvature correction coeff p0')
        flatHDU.header['COEF1_1'] = (self.CF.coeff[1], 'Curvature correction coeff p1')
        flatHDU.header['COEF1_2'] = (self.CF.coeff[2], 'Curvature correction coeff p2')
        flatHDU.header['COEF2_0'] = (self.CF.coeff2[0], '2nd Curvature correction coeff p0')
        flatHDU.header['COEF2_1'] = (self.CF.coeff2[1], '2nd Curvature correction coeff p1')
        flatHDU.header['COEF2_2'] = (self.CF.coeff2[2], '2nd Curvature correction coeff p2')
        flatHDU.header['CCDNAME'] = (self.h['CCDNAME'], 'Prodctname of CCD')

        try:
            flatHDU.header['WAVELEN'] = (self.h['WAVELEN'], 'Angstrom')
        except:
            pass
        try:
            flatHDU.header['GRATWVLN'] = (self.h['GRATWVLN'], 'Angstrom')
        except:
            pass
        for comment in self.h['COMMENT']:
            flatHDU.header.add_comment(comment)

        flatHDU.header.add_comment('Tilt Corrected')
        flatHDU.header.add_comment('1st Curvature Corrected')
        flatHDU.header.add_comment('2nd Curvature Corrected')
        if self.yFringe is not None:
            flatHDU.header.add_comment('y-dir Fringe Subtractd')
        if self.xFringe is not None:
            flatHDU.header.add_comment('x-dir Fringe Subtractd')

        flatHDU.writeto(join(self.pcaldir, flatName), overwrite=True)

        if self.yFringe is not None:
            self.log += "> Save y-dir Fringe Pattern.<br>"
            self._writeLog()
            yf = 10 ** self.yFringe
            yFringeHDU = fits.PrimaryHDU(yf)
            yFringeHDU.header['EXPTIME'] = (self.h['EXPTIME'], 'Second')
            yFringeHDU.header['OBSTIME'] = (obstime, 'Observation Time (UT)')
            yFringeHDU.header['DATE'] = (self.h['DATE'], 'File Creation Date (UT)')
            yFringeHDU.header['FILTMIN'] = (self.yf_min, 'Minimum Wavelet Filtering Range')
            yFringeHDU.header['FILTMAX'] = (self.yf_max, 'Maximum Wavelet Filtering Range')
            yFringeHDU.header.add_comment('Tilt Corrected')
            yFringeHDU.header.add_comment('1st Curvature Corrected')
            yFringeHDU.writeto(join(self.pcaldir, yFringeName), overwrite=True)
        if self.xFringe is not None:
            self.log += "> Save x-dir Fringe Pattern.<br>"
            self._writeLog()
            xf = 10 ** self.xFringe
            xFringeHDU = fits.PrimaryHDU(xf)
            xFringeHDU.header['EXPTIME'] = (self.h['EXPTIME'], 'Second')
            xFringeHDU.header['OBSTIME'] = (obstime, 'Observation Time (UT)')
            xFringeHDU.header['DATE'] = (self.h['DATE'], 'File Creation Date (UT)')
            xFringeHDU.header['FILTMIN'] = (self.xf_min, 'Minimum Wavelet Filtering Range')
            xFringeHDU.header['FILTMAX'] = (self.xf_max, 'Maximum Wavelet Filtering Range')
            xFringeHDU.header.add_comment('Tilt Corrected')
            xFringeHDU.header.add_comment('1st Curvature Corrected')
            xFringeHDU.writeto(join(self.pcaldir, xFringeName), overwrite=True)

        self.fflatL.pop(self.fidx)
        self.fflatGBL.pop(self.fidx)

        if len(self.fflatL) != 0:
            self.B_s6_yes.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")
        else:
            self.B_s6_yes.setStyleSheet(f"background-color: {self.bg_second}; color:{self.font_normal};")
            self.B_s6_No.setStyleSheet(f"background-color: {self.btn_1}; color:{self.bg_primary};")

        qSleep(0.01)
        self.List_VL[6].setGeometry(QtCore.QRect(10,85,280,350))
        
        self.log += "> Done.<br>"
        self._writeLog()
        qSleep(0.2)
        self.fig.canvas.draw_idle()

    def s6_yes(self):
        self.Initialize()
        self.B_s6_yes.setStyleSheet(f"background-color: {self.bg_second};")
        self.step(0)
        qSleep(0.1)

    def Initialize(self):
        self.subStepNum = -1
        self.xFringe = None
        self.yFringe = None
        self.s1 = None
        self.s2 = None
        self.p_s5_ori = None
        self.p_s5_ff = None
        self.im_s5 = None
        self.frameNum = 3
        self.LE_s1_tilt.setText('0')
        self.LE_s2_p0.setText('0')
        self.LE_s2_p1.setText('0')
        self.LE_s2_p2.setText('0')
        self.LE_s4_p0.setText('0')
        self.LE_s4_p1.setText('0')
        self.LE_s4_p2.setText('0')
        self.imFx = None
        self.imFy = None
        self.im_s3_3 = None

        self.CB_s0_fflist.removeItem(self.fidx)
        self.fidx = 0
        self.CB_s0_fflist.setCurrentIndex(self.fidx)

        self.B_s3_wvShow.setEnabled(False)
        self.B_s3_2_FRapply.setEnabled(False)
        self.B_s3_2_simple.setEnabled(False)
        self.B_s3_2_FringeShow.setEnabled(False)
        self.B_s3_2_resShow.setEnabled(False)
        self.B_s3_2_blink.setEnabled(False)

        self.B_s3_3_Blink.setEnabled(False)

        self.B_s3_4_wvShow.setEnabled(False)
        self.B_s3_4_FRapply.setEnabled(False)
        self.B_s3_4_simple.setEnabled(False)
        self.B_s3_4_FringeShow.setEnabled(False)
        self.B_s3_4_resShow.setEnabled(False)
        self.B_s3_4_blink.setEnabled(False)

        self.B_s1_Apply.setEnabled(False)
        self.B_s2_Apply.setEnabled(False)
        self.B_s4_Apply.setEnabled(False)
        self.B_s5_Show.setEnabled(False)
        self.B_s5_cFlat.setEnabled(False)
        self.B_s5_Blink.setEnabled(False)
        self.B_s5_msFlat.setEnabled(False)
        
        for i, axl in enumerate(self.ax):
            if not i == 3:
                for j, ax in enumerate(axl):
                    ax.cla()
                    if not i == 7:
                        ax.set_position(self.ax_pos[i][j])

        for i, axls in enumerate(self.ax_sub):
            for j, ax in enumerate(axls):
                ax.cla()
                ax.set_position(self.ax_sub_pos[i][j])
        self.chFlat()
        self.fig.canvas.draw_idle()

    def s6_no(self):
        self.Next()
        qSleep(0.1)

    def s7_proc(self):
        self.log = "> Run Preprocess.<br>"
        self._writeLog()
        self.runCam = self.CB_s7_camlist.currentIndex()
        self.runTarget = self.CB_s7_targetlist.currentIndex()

        lTarget = glob(join(self.rawdir, '*'))
        lTarget.sort()
        lBand = ['A', 'B']
        cmRaster = [cm.ha, cm.ca]
        lfFlat_A = glob(join(self.pcaldir, 'FISS_FLAT*A.fts'))
        lfFlat_A.sort()
        lfFlat_B = glob(join(self.pcaldir, 'FISS_FLAT*B.fts'))
        lfFlat_B.sort()
        tlfFlat = [lfFlat_A, lfFlat_B]
        lfXF_A = glob(join(self.pcaldir, 'FISS_xFringe*A.fts'))
        lfXF_B = glob(join(self.pcaldir, 'FISS_xFringe*B.fts'))
        lfXF_A.sort()
        lfXF_B.sort()
        tlfXF = [lfXF_A, lfXF_B]
        lfYF_A = glob(join(self.pcaldir, 'FISS_yFringe*A.fts'))
        lfYF_B = glob(join(self.pcaldir, 'FISS_yFringe*B.fts'))
        lfYF_A.sort()
        lfYF_B.sort()
        tlfYF = [lfYF_A, lfYF_B]
        init = True

        # read Flat and Fringe
        tFlatHeader = [None]*2
        tFlat = [None]*2
        tFlatJD = [None]*2
        tXF = [None]*2
        tXFJD = [None]*2
        tYF = [None]*2
        tYFpks = [None]*2
        tYFaws = [None]*2
        tYFJD = [None]*2
        tSP = [None]*2
        for i, lfF in enumerate(tlfFlat):
            # read Flat
            nFlat = len(lfF)
            lh = [None]*nFlat
            lJD = np.zeros(nFlat)
            init = True
            for j,f in enumerate(lfF):
                opn = fits.open(f)[0]
                if init:
                    ny, nw = opn.data.shape
                    lFlat = np.zeros((nFlat, ny, nw), dtype=float)
                    init = False
                lh[j] = opn.header
                lFlat[j] = opn.data
                lJD[j] = Time(proc_base.fname2isot(f)).jd

            tFlat[i] = lFlat
            tFlatHeader[i] = lh
            tFlatJD[i] = lJD

            # read xFringe
            lfX = tlfXF[i]
            init = True
            if len(lfX):
                lXFJD = np.zeros(len(lfX))
                for j,f in enumerate(lfX):
                    xf = fits.getdata(f)
                    if init:
                        nf, ny, nw = xf.shape
                        lXF = np.zeros((len(lfX), nf, ny, nw))
                        init = False
                    lXF[j] = xf
                    lXFJD[j] = Time(proc_base.fname2isot(f)).jd
                tXF[i] = lXF
                tXFJD[i] = lXFJD
                    
            # read yFringe
            lfY = tlfYF[i]
            init = True
            if len(lfY):
                lpks = [None]*len(lfY)
                lYFaws = [None]*len(lfY)
                lSP = [None]*len(lfY)
                lYFJD = np.zeros(len(lfY))
                h = fits.getheader(lfY[0])
                yy = int(h['date'][:4])
                for j,f in enumerate(lfY):
                    yf = fits.getdata(f)
                    if init:
                        nf, ny, nw = yf.shape
                        lYF = np.zeros((len(lfY), nf, ny, nw))
                        init = False
                    lYF[j] = yf
                    # get slit pattern position
                    lyf = np.log10(yf)
                    d2y = np.gradient(np.gradient(lyf[nf//2],axis=0), axis=0).mean(1)
                    pks = find_peaks(d2y[5:-5], d2y[5:-5].std())[0]+5
                    lpks[j] = pks
                    if yy >= 2023:
                        sp = proc_base.yf2sp(lyf.mean(0))
                    else:
                        sp = proc_base.yf2sp(lyf[nf//2])
                    sp -= sp[5:-5,5:-5].mean()

                    # get wavelet
                    # wvl = Wavelet(lyf[nf//2], dt=1, axis=0)
                    wvl = Wavelet(lyf[nf//2] - sp, dt=1, axis=0)
                    lYFaws[j] = np.abs(wvl.wavelet)
                    lYFJD[j] = Time(proc_base.fname2isot(f)).jd
                    lSP[j] = 10**sp
                tSP[i] = lSP
                tYF[i] = lYF
                tYFJD[i] = lYFJD
                tYFpks[i] = lpks
                tYFaws[i] = lYFaws
                    
        
        for kk, dTarget in enumerate(lTarget):
            if self.runTarget:
                if kk != self.runTarget -1:
                    continue

            if self.stop:
                break
            self.log += f"> Run for {basename(dTarget)} directory.<br>"
            self._writeLog()

            for idx, band in enumerate(lBand):
                if self.runCam == 1 and idx == 1:
                    continue
                elif self.runCam == 2 and idx == 0:
                    continue
                lf = glob(join(dTarget, f'*_{band}*.fts'))
                lf.sort()
                nlf = len(lf)
                if self.stop:
                    break
                chclim = False
                TXTinit = True
                if self.p_s7_prof is not None:
                    self.p_s7_prof.remove()
                    self.ax[7][4].cla()
                    self.im_s7_spec.remove()
                    self.im_s7_R1.remove()
                    self.im_s7_R2.remove()
                    self.im_s7_R3.remove()
                    self.im_s7_R4.remove()
                    self.p_s7_prof = None
                    qSleep(1)

                xfID = np.arange(len(tlfXF[idx]), dtype=int)
                yfID = np.arange(len(tlfYF[idx]), dtype=int)
                if len(tlfFlat[idx]) and nlf:
                    self.log += f"> Run for cam {band}.<br>"
                    self._writeLog()

                    for i, f in enumerate(lf):
                        if self.stop:
                            break

                        if TXTinit:
                            self.log += f"> Run {i+1}/{nlf}.<br>"
                            TXTinit = False
                        else:
                            self.log = self.log.replace(f"{i}/{nlf}", f"{i+1}/{nlf}")

                        self._writeLog()

                        if f.find('BiasDark') > -1:
                            db = fits.getdata(f)
                            chclim = True
                            continue
                        # if i != 27:
                        #     continue
                        opn = fits.open(f)[0]
                        data = opn.data
                        h = opn.header
                        jd = Time(h['date']).jd
                        wh = np.abs(jd - tFlatJD[idx]).argmin()
                        flat = tFlat[idx][wh]
                        fjd = tFlatJD[idx][wh]
                        ch = tFlatHeader[idx][wh]
                        tilt = ch['tilt']
                        p1_0 = ch['coef1_0']
                        p1_1 = ch['coef1_1']
                        p1_2 = ch['coef1_2']
                        p2_0 = ch['coef2_0']
                        p2_1 = ch['coef2_1']
                        p2_2 = ch['coef2_2']

                        data1 = data-db

                        td = proc_base.tilt_correction(data1, tilt, cubic=True)
                        cd1 = proc_base.curvature_correction(td, [p1_0, p1_1, p1_2])
                        
                        if len(tlfYF[idx]):
                            why = yfID[np.abs(tYFJD[idx] - fjd)*24*3600 < 10][0]
                            
                            # image shift correction
                            self.sp = tSP[idx][why]

                            self.testRaw = cd1.copy()
                            self.pks = tYFpks[idx][why]
                            if len(self.pks):
                                ssp, spks = proc_base.calShift(cd1, self.sp, self.pks)
                                self.spks = spks
                                cd1 /= ssp
                                rsp = proc_base.raw2sp(cd1, spks)
                                self.rsp = rsp
                                cd1 /= rsp
                            else:
                                cd1 /= self.sp 
                            
                            ryf = proc_base.rawYF(cd1, tYFaws[idx][why])
                            self.ryf = ryf
                            cd1 /= ryf

                            # cd1 /= tYF[idx][why][3]
                            self.cd1 = cd1


                        if len(tlfXF[idx]):
                            whx = xfID[np.abs(tXFJD[idx] - fjd)*24*3600 < 10][0]
                            XF = tXF[idx][whx]
                            cd1 /= XF[nf//2]

                        cd2 = proc_base.curvature_correction(cd1, [p2_0, p2_1, p2_2])
                        cd2 /= flat
                        cd2 = cd2[:,5:-5,5:-5].astype('int16')
                        self.cd2 = cd2
                        shape = cd2.shape

                        if self.p_s7_prof is None:
                            p0 = int(ch['crpix1']-5)
                            p_4 = int(p0 - 4/ch['cdelt1'])
                            self.ax[7][0].set_axis_off()
                            self.ax[7][1].set_axis_off()
                            self.ax[7][2].set_axis_off()
                            self.ax[7][3].set_axis_off()
                            self.ax[7][0].set_title('-4.0 $\\AA$')
                            self.ax[7][2].set_title('+0.0 $\\AA$')
                            self.ax[7][5].set_title(f'Spectrogram (x={shape[0]//2})')
                            self.ax[7][5].set_xlabel('Wavelength (pix)')
                            self.ax[7][5].set_ylabel('Slit (pix)')

                            self.ax[7][4].set_xlabel('Wavelength (pix)')
                            self.ax[7][4].set_ylabel('Intensity (DN)')
                            if idx == 0:
                                p0_5 = int(p0 + 0.7/ch['cdelt1'])
                                p_0_5 = int(p0 - 0.7/ch['cdelt1'])
                                self.ax[7][1].set_title('-0.7 $\\AA$')
                                self.ax[7][3].set_title('+0.7 $\\AA$')
                            else:
                                p0_5 = int(p0 + 0.5/ch['cdelt1'])
                                p_0_5 = int(p0 - 0.5/ch['cdelt1'])
                                self.ax[7][1].set_title('-0.5 $\\AA$')
                                self.ax[7][3].set_title('+0.5 $\\AA$')
                            self.p_s7_prof = self.ax[7][4].plot(cd2[shape[0]//2,shape[1]//2])[0]
                            self.ax[7][4].set_xlim(-0.5, shape[2]-0.5)
                            self.im_s7_spec = self.ax[7][5].imshow(cd2[shape[0]//2], cmRaster[idx], origin='lower')

                            self.im_s7_R1 = self.ax[7][0].imshow(cd2[:,:,p_4].T, cmRaster[idx], origin='lower')

                            self.im_s7_R2 = self.ax[7][1].imshow(cd2[:,:,p_0_5].T, cmRaster[idx], origin='lower')
                            m = cd2[:,:,p_0_5].mean()
                            std = cd2[:,:,p_0_5].std()
                            self.im_s7_R2.set_clim(m-std*2,m+std*2)

                            self.im_s7_R3 = self.ax[7][2].imshow(cd2[:,:,p0].T, cmRaster[idx], origin='lower')
                            m = cd2[:,:,p0].mean()
                            std = cd2[:,:,p0].std()
                            self.im_s7_R3.set_clim(m-std*2,m+std*2)

                            self.im_s7_R4 = self.ax[7][3].imshow(cd2[:,:,p0_5].T, cmRaster[idx], origin='lower')
                            m = cd2[:,:,p0_5].mean()
                            std = cd2[:,:,p0_5].std()
                            self.im_s7_R4.set_clim(m-std*2,m+std*2)
                            init = False
                        else:
                            
                            self.p_s7_prof.set_ydata(cd2[shape[0]//2,shape[1]//2])
                            self.im_s7_spec.set_data(cd2[shape[0]//2])
                            self.im_s7_R1.set_data(cd2[:,:,p_4].T)
                            self.im_s7_R2.set_data(cd2[:,:,p_0_5].T)
                            self.im_s7_R3.set_data(cd2[:,:,p0].T)
                            self.im_s7_R4.set_data(cd2[:,:,p0_5].T)

                            if chclim:
                                prof = cd2[shape[0]//2,shape[1]//2]
                                self.ax[7][4].set_ylim(prof.min()*0.95, prof.max()*1.05)
                                self.im_s7_spec.set_clim(cd2[shape[0]//2].min(), cd2[shape[0]//2].max())
                                
                                self.im_s7_R1.set_clim(cd2[:,:,p_4].min(), cd2[:,:,p_4].max())
                                m = cd2[:,:,p_0_5].mean()
                                std = cd2[:,:,p_0_5].std()
                                self.im_s7_R2.set_clim(m-std*2,m+std*2)
                                m = cd2[:,:,p0].mean()
                                std = cd2[:,:,p0].std()
                                self.im_s7_R3.set_clim(m-std*2,m+std*2)
                                m = cd2[:,:,p0_5].mean()
                                std = cd2[:,:,p0_5].std()
                                self.im_s7_R4.set_clim(m-std*2,m+std*2)

                                self.im_s7_R1.set_extent([-0.5, shape[0]-0.5, -0.5, shape[1]-0.5])
                                self.im_s7_R2.set_extent([-0.5, shape[0]-0.5, -0.5, shape[1]-0.5])
                                self.im_s7_R3.set_extent([-0.5, shape[0]-0.5, -0.5, shape[1]-0.5])
                                self.im_s7_R4.set_extent([-0.5, shape[0]-0.5, -0.5, shape[1]-0.5])
                                self.ax[7][0].set_xlim(-0.5, shape[0]-0.5)
                                self._writeLog()
                                chclim = False
                            
                            
                        self.ax[7][4].set_title(f'Profile ({i+1}/{nlf})')
                        
                        
                        self.fig.canvas.draw_idle()
                        qSleep(0.2)
                        # self.fig.canvas.draw_idle()
                        qSleep(0.1)

                        # save fits
                        fn = basename(f)
                        fn = fn.replace(f'{band}.fts', f'{band}1.fts')
                        dname = join(self.procdir, basename(dTarget))
                        if not isdir(dname):
                            makedirs(dname)
                        fn = join(dname, fn)
                        
                        if h['STRTIME'].find('.') < 10:
                            h['STRTIME'] = h['STRTIME'].replace('-', 'T').replace('.', '-')
                        if h['ENDTIME'].find('.') < 10:
                            h['ENDTIME'] = h['ENDTIME'].replace('-', 'T').replace('.', '-')
                        obstime = (Time(h['STRTIME']).jd + Time(h['ENDTIME']).jd)/2
                        obstime = Time(obstime, format='jd').isot

                        hdu = fits.PrimaryHDU(cd2)
                        hdu.header['CRPIX1'] = (ch['crpix1']-5, 'reference pixel position')
                        hdu.header['CDELT1'] = (ch['cdelt1'], 'angstrom/pixel')
                        hdu.header['CRVAL1'] = (ch['crval1'], 'reference wavelength (angstrom)')
                        hdu.header['EXPTIME'] = (h['EXPTIME'], 'Second')
                        hdu.header['OBSTIME'] = (obstime, 'Observation Time (UT)')
                        hdu.header['DATE'] = (h['DATE'], 'File Creation Date (UT)')
                        hdu.header['STRTIME'] = (h['STRTIME'], 'Scan Start Time')
                        hdu.header['ENDTIME'] = (h['ENDTIME'], 'Scan Finish Time')
                        hdu.header['HBINNING'] = (h['HBINNING'], 'Horizontal Binning')
                        hdu.header['VBINNING'] = (h['VBINNING'], 'Vertical Binning')
                        hdu.header['OBSERVER'] = (h['OBSERVER'], 'The name of main observer')
                        hdu.header['TEL_XPOS'] = (h['TEL_XPOS'], 'X position of Telescope on the Sun')
                        hdu.header['TEL_YPOS'] = (h['TEL_YPOS'], 'X position of Telescope on the Sun')
                        hdu.header['TARGET'] = (h['TARGET'], 'Observation Target')
                        hdu.header['TILT'] = (ch['tilt'], 'Degree')
                        hdu.header['COEF1_0'] = (ch['coef1_0'], 'Curvature correction coeff p0')
                        hdu.header['COEF1_1'] = (ch['coef1_1'], 'Curvature correction coeff p1')
                        hdu.header['COEF1_2'] = (ch['coef1_2'], 'Curvature correction coeff p2')
                        hdu.header['COEF2_0'] = (ch['coef2_0'], '2nd Curvature correction coeff p0')
                        hdu.header['COEF2_1'] = (ch['coef2_1'], '2nd Curvature correction coeff p1')
                        hdu.header['COEF2_2'] = (ch['coef2_2'], '2nd Curvature correction coeff p2')
                        hdu.header['CCDNAME'] = (ch['CCDNAME'], 'Prodctname of CCD')
                        try:
                            hdu.header['WAVELEN'] = (ch['WAVELEN'], 'Angstrom')
                        except:
                            pass
                        try:
                            hdu.header['GRATWVLN'] = (ch['GRATWVLN'], 'Angstrom')
                        except:
                            pass
                        hdu.header['GRATSTEP'] = (h['GRATSTEP'], 'Grating step count')
                        hdu.header['GRATANGL'] = (h['GRATANGL'], 'Grating angle')
                        hdu.header['CCDTEMP'] = (h['CCDTEMP'], 'Cooling Temperature of CCD')
                        hdu.header['STEPSIZE'] = (h['STEPSIZE'], 'Step size to move scanner (um)')
                        hdu.header['STEPTIME'] = (h['STEPTIME'], 'Step duration time (ms)')
                        hdu.header['ELAPTIME'] = (h['ELAPTIME'], 'Elapse time during scanning (s)')
                        hdu.header['PAMPGAIN'] = (h['PAMPGAIN'], 'Value Range: 0-2')
                        hdu.header['EMGAIN'] = (h['EMGAIN'], 'Value Range: 0-255')
                        hdu.header.add_comment('Tilt Corrected')
                        hdu.header.add_comment('1st Curvature Corrected')
                        hdu.header.add_comment('2nd Curvature Corrected')
                        if len(tlfYF[idx]):
                            hdu.header.add_comment('y-dir Fringe Subtractd')
                        if len(tlfXF[idx]):
                            hdu.header.add_comment('x-dir Fringe Subtractd')
                        hdu.writeto(fn, overwrite=True)

                    self.log += f"> {band} Done.<br>"
                    self._writeLog()

        if self.stop:
            self.log += "> Stop.<br>"
        else:
            self.log += "> Done.<br>"
        self._writeLog()
        self.stop = False
        print('Done')
                    
    def s7_comp(self):
        self.log = "> Run PCA Compression.<br>"
        self._writeLog()
        maxnum = 10
        self.ax_hide()
        self.ax_s7_comp.set_visible(True)
        self.fig.canvas.draw_idle()
        qSleep(0.05)

        self.runCam = self.CB_s7_camlist.currentIndex()
        self.runTarget = self.CB_s7_targetlist.currentIndex()

        lBand = ['A', 'B']
        ltol = [1e-1, 5e-1]

        # Target dir
        lTarget = glob(join(self.rawdir, '*'))
        lTarget.sort()

        for kk, Target in enumerate(lTarget):
            if self.runTarget:
                if kk != self.runTarget -1:
                    continue

            if self.stop:
                break

            self.log += f"> Run for {basename(Target)} directory.<br>"
            self._writeLog()

            makePfile = True
            num = 0
            for bb, band in enumerate(lBand):
                if self.runCam == 1 and bb == 1:
                    continue
                elif self.runCam == 2 and bb == 0:
                    continue
                if self.stop:
                    break

                if self.p_s7_comp is not None:
                    self.p_s7_comp.remove()
                    self.p_s7_comp = None
                    self.p_s7_odata.remove()
                
                tol = ltol[bb]
                lf = glob(join(Target, f'*_{band}*.fts'))
                lf.sort()
                nlf = len(lf)
                TXTinit = True
                if nlf:
                    self.log += f"> Run for cam {band}.<br>"
                    self._writeLog()

                    for i, f in enumerate(lf):
                        if self.stop:
                            break

                        if TXTinit:
                            self.log += f"> Run {i+1}/{nlf}.<br>"
                            TXTinit = False
                        else:
                            self.log = self.log.replace(f"{i}/{nlf}", f"{i+1}/{nlf}")

                        self._writeLog()
                        # if i != 27:
                        #     continue
                        if f.find('BiasDark') > -1:
                            makePfile = True
                            num = 0
                            continue
                        if num >= maxnum:
                            num = 0
                            makePfile = True
                        f = f.replace('raw', 'proc')
                        f = f.replace('.fts', '1.fts')

                        if makePfile:
                            Evec, spec, odata, ev = proc_base.PCA_compression(f, ret=True, tol=tol)
                            ncoeff = Evec.shape[0]
                            # if ncoeff == 50:
                            #     maxnum = 10
                            self.log += f"> ncoeff: {ncoeff}.<br> Eval:{ev:.3f} <br>"
                            self._writeLog()
                            makePfile = False
                            pfile = f.replace('.fts', '_p.fts')
                            pfile = pfile.replace('proc', 'comp')
                            nx, ny, nw = odata.shape
                            if self.p_s7_comp is None:
                                self.p_s7_odata = self.ax_s7_comp.plot(odata[nx//2, ny//2], label='proc', color='C0')[0]
                                self.p_s7_comp = self.ax_s7_comp.plot(spec[nx//2, ny//2], label='comp', color='C1')[0]
                                self.ax_s7_comp.legend()
                                self.ax_s7_comp.set_xlim(0, nw-1)
                                self.ax_s7_comp.set_xlabel('Wavelength (pix)')
                                self.ax_s7_comp.set_ylabel('Intensity (DN)')
                        else:
                            res = proc_base.PCA_compression(f, Evec=Evec, pfile=pfile, tol=tol, ret=True)
                            odata = res[2]
                            spec = res[1]
                        self.p_s7_odata.set_ydata(odata[nx//2, ny//2])
                        self.p_s7_comp.set_ydata(spec[nx//2, ny//2])
                        self.ax_s7_comp.set_ylim(odata[nx//2, ny//2].min()*0.98, odata[nx//2, ny//2].max()*1.02)
                        self.ax_s7_comp.set_title(f"Profile ({i+1}/{nlf})")
                        self.fig.canvas.draw_idle()
                        qSleep(0.1)
                            
                        num += 1 

                qSleep(5)
        if self.stop:
            self.log += "> Stop.<br>"
        else:
            self.log += "> Done.<br>"
        self._writeLog()
        self.ax_s7_comp.set_visible(False)
        self.stop = False
        print('Done')

    def s7_stop(self):
        self.stop = True
