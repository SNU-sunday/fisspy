from PyQt5 import QtWidgets, QtCore, QtGui
from os.path import join, dirname, basename
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fisspy.preprocess import proc_base
from fisspy.analysis.wavelet import Wavelet
from glob import glob
from astropy.io import fits
from scipy.optimize import curve_fit

class prepGUI:
    def __init__(self, basedir, ffocA=None, ffocB=None):
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
        self.procdir = join(basedir, 'proc')
        self.compdir = join(basedir, 'comp')
        self.rawdir = join(basedir, 'raw')
        self.ffocA = ffocA
        self.ffocB = ffocB
        self.fflatL = glob(join(self.rcaldir, '*_Flat.fts'))
        self.fflatGBL = [None] * len(self.fflatL)
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
                          "Step 2: Curvature Correction",
                          "Step 3: Fringe Subtraction",
                          "Step 4: Re-curve Correction",
                          "Step 5: Make Flat",
                          "Step 6: Check Flat",
                          "Step 7: Another Flat ?",
                          "Step 8: Run Preproccess",
                          "Step 9: PCA Compression"]
        
        self.List_subStep = ["3-1: Atlas Subtraction",
                             "3-2: Vertical Fringe",
                             "3-3: Spectrum Mask",
                             "3-4: Horizontal Frigne"]
        
        self.nStep = len(self.List_step)
        self.nsubStep = len(self.List_subStep)
        
        self.fig = plt.figure(figsize=[17,7])

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

        self.ax = [[ax_s0], [ax_s1_1, ax_s1_2], [ax_s2_1, ax_s2_2, ax_s2_3]]
        
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
        # self.fig.canvas.mpl_connect()
        self.fig.show()

    def ax_hide(self):
        for lax in self.ax:
            for ax in lax:
                ax.set_visible(False)
        for lax in self.ax_sub:
            for ax in lax:
                ax.set_visible(False)

    def step(self, num):
        h = self.fig.get_figheight()
        self.ax_hide()
        if self.stepNum != 3:
            for wg in self.StepWidgets[self.stepNum]:
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
                self.s1_data = 10**self.CF.logRF[3]
        self.List_VL[self.stepNum].setGeometry(QtCore.QRect(10,85,280,350))
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
        
        self.subStepNum = num
        self.fig.canvas.draw_idle()
        self.List_subVL[num].setGeometry(QtCore.QRect(10,85,280,350))
        self.fig.set_figheight(h)

    def _onKey(self, event):
        None

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
        self.panel.setMaximumSize(QtCore.QSize(300, h*100-22))
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
        self.StepWidgets = [None]*4#self.nStep
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

            # add widgets
            self.VL_s0.addWidget(self.L_s0_fflist)
            self.VL_s0.addWidget(self.CB_s0_fflist)
            self.VL_s0.addLayout(self.HL_s0_frame)
            self.vboxCtrl.addLayout(self.VL_s0)

            self.StepWidgets[0] = [self.L_s0_fflist, self.CB_s0_fflist, self.L_s0_frame, self.LE_s0_frame, self.L_s0_nframe, self.B_s0_pf, self.B_s0_nf]     
            
        # create Step1 tilt Widget
        if True:
            self.VL_s1 = QtWidgets.QVBoxLayout()
            self.HL0_s1 = QtWidgets.QHBoxLayout()
            self.L_s1_get_tilt = QtWidgets.QLabel()
            self.L_s1_get_tilt.setText("Get Tilt: ")
            self.L_s1_get_tilt.setFont(self.fNormal)

            self.B_s1_run = QtWidgets.QPushButton()
            self.B_s1_run.setText("Auto Run")
            self.B_s1_run.setFont(self.fNormal)
            self.B_s1_run.setStyleSheet(f"background-color: {self.bg_second};")
            
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
            self.B_s2_run.setStyleSheet(f"background-color: {self.bg_second};")

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

            self.HL_s2_coeff = QtWidgets.QHBoxLayout()

            
            self.HL_s2_coeff.addWidget(self.L_s2_p0)
            self.HL_s2_coeff.addWidget(self.LE_s2_p0)
            self.HL_s2_coeff.addWidget(self.L_s2_p1)
            self.HL_s2_coeff.addWidget(self.LE_s2_p1)
            self.HL_s2_coeff.addWidget(self.L_s2_p2)
            self.HL_s2_coeff.addWidget(self.LE_s2_p2)
            self.VL_s2.addLayout(self.HL0_s2)
            self.VL_s2.addLayout(self.HL_s2_coeff)
            self.VL_s2.addWidget(self.B_s2_Apply)

            self.vboxCtrl.addLayout(self.VL_s2)

            self.StepWidgets[2] = [self.L_s2_get_curve, self.B_s2_run, self.L_s2_p0, self.LE_s2_p0, self.L_s2_p1, self.LE_s2_p1, self.L_s2_p2, self.LE_s2_p2, self.B_s2_Apply]

            self.B_s2_run.clicked.connect(self.s2_Run)
            self.B_s2_Apply.clicked.connect(self.s2_Apply)

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
            self.B_s3_1_run.setStyleSheet(f"background-color: {self.bg_second};")

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

        # create Step3-2
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
            self.B_s3_wvCal.setStyleSheet(f"background-color: {self.bg_second};")
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

            self.yf_max = 115
            self.L_s3_2_FRmax = QtWidgets.QLabel()
            self.L_s3_2_FRmax.setText("max:")
            self.L_s3_2_FRmax.setFont(self.fNormal)
            self.LE_s3_2_FRmax = QtWidgets.QLineEdit()
            self.LE_s3_2_FRmax.setText("115")
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
            self.B_s3_2_simple.setStyleSheet(f"background-color: {self.bg_second};")
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

        # create Step3-3
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
            self.L_s3_3_GMW.setText("Get mask width: ")
            self.L_s3_3_GMW.setFont(self.fNormal)

            self.B_s3_3_GMW = QtWidgets.QPushButton()
            self.B_s3_3_GMW.setText("Run")
            self.B_s3_3_GMW.setFont(self.fNormal)
            self.B_s3_3_GMW.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_3_GMW.clicked.connect(self.s3_3_Run)

            self.HL_s3_3_GMW.addWidget(self.L_s3_3_GMW)
            self.HL_s3_3_GMW.addWidget(self.B_s3_3_GMW)

            # msk width
            self.HL_s3_3_MW = QtWidgets.QHBoxLayout()
            self.L_s3_3_MW = QtWidgets.QLabel()
            self.L_s3_3_MW.setText("Mask width: ")
            self.L_s3_3_MW.setFont(self.fNormal)

            self.LE_s3_3_MW = QtWidgets.QLineEdit()
            self.LE_s3_3_MW.setText("0")
            self.LE_s3_3_MW.setStyleSheet(f"background-color: {self.bg_second}; border: 1px solid {self.font_normal};")
            self.LE_s3_3_MW.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)

            self.B_s3_3_Apply = QtWidgets.QPushButton()
            self.B_s3_3_Apply.setText("Apply")
            self.B_s3_3_Apply.setFont(self.fNormal)
            self.B_s3_3_Apply.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_3_Apply.clicked.connect(self.s3_3_Apply)

            self.HL_s3_3_MW.addWidget(self.L_s3_3_MW)
            self.HL_s3_3_MW.addWidget(self.LE_s3_3_MW)
            self.HL_s3_3_MW.addWidget(self.B_s3_3_Apply)

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
            self.VL_s3_3.addLayout(self.HL_s3_3_MW)
            self.VL_s3_3.addLayout(self.HL_s3_3_show)
            self.VL_s3_3.addLayout(self.HL_s3_3_frame)

            self.vboxCtrl.addLayout(self.VL_s3_3)
            self.subStepWidgets[2] = [self.L_s3_3_subStep, self.L_s3_3_GMW, self.B_s3_3_GMW, self.L_s3_3_MW, self.LE_s3_3_MW, self.B_s3_3_Apply, self.L_s3_3_res, self.B_s3_3_Blink, self.L_s3_3_frame, self.LE_s3_3_frame, self.L_s3_3_nframe, self.B_s3_3_pf, self.B_s3_3_nf]

            self.im_s3_3 = None
        
        # create Step3-4
        if True:
            self.VL_s3_4 = QtWidgets.QVBoxLayout()
            self.L_s3_4_subStep = QtWidgets.QLabel()
            self.L_s3_4_subStep.setText(self.List_subStep[1])
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
            self.B_s3_4_wvCal.setStyleSheet(f"background-color: {self.bg_second};")
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
            self.B_s3_4_simple.setStyleSheet(f"background-color: {self.bg_second};")
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
            self.VL_s3_4.addLayout(self.HL_s3_4_frame)

            self.vboxCtrl.addLayout(self.VL_s3_4)

            self.subStepWidgets[3] = [self.L_s3_4_subStep, self.L_s3_4_wvlet, self.B_s3_4_wvCal, self.B_s3_4_wvShow, self.L_s3_4_FR, self.L_s3_4_FRmin, self.LE_s3_4_FRmin, self.L_s3_4_FRmax, self.LE_s3_4_FRmax, self.B_s3_4_FRapply, self.L_s3_4_calFringe, self.B_s3_4_simple, self.B_s3_4_gauss, self.B_s3_4_FringeShow, self.L_s3_4_res, self.B_s3_4_resShow, self.B_s3_4_blink, self.L_s3_4_frame, self.LE_s3_4_frame, self.L_s3_4_nframe, self.B_s3_4_pf, self.B_s3_4_nf]
            self.imFx = None


        self.List_VL = [self.VL_s0, self.VL_s1, self.VL_s2]
        self.List_subVL = [self.VL_s3_1, self.VL_s3_2, self.VL_s3_3, self.VL_s3_4]

        for vl in self.List_VL:
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
        
        self.fidx = self.CB_s0_fflist.currentIndex()
        f = self.fflatL[self.fidx]
        n = f.find('A_Flat')
        if n != -1:
            ffoc = self.ffocA
        else:
            ffoc = self.ffocB
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
        plt.pause(0.01)
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
            self.subStep(self.subStepNum+1)
        elif self.stepNum == 3 and self.subStepNum == self.nsubStep-1:
            for wg in self.subStepWidgets[self.stepNum]:
                wg.setVisible(False)
            self.step(self.stepNum+1)
        elif self.stepNum < self.nStep-1:
            self.step(self.stepNum+1)
        else:
            self.stepNum == self.nStep-1
        

    def Prev(self):
        if self.stepNum <= 0:
            self.stepNum = 0
        elif self.stepNum == 4:
            for wg in self.StepWidgets[self.stepNum]:
                wg.setVisible(False)
            self.stepNum -= 1
            self.subStep(self.subStepNum-1)
        elif self.stepNum == 3 and self.subStepNum == 0:
            for wg in self.subStepWidgets[self.subStepNum]:
                wg.setVisible(False)
            self.subStepNum = -1
            self.step(self.stepNum-1)
        elif self.stepNum == 3 and self.subStepNum > 0:
            self.subStep(self.subStepNum-1)
        else:
            self.step(self.stepNum-1)

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

    def s1_img(self, img):
        self.log += f"> Tilt: {self.CF.tilt:.3f} degree.<br>"
        self._writeLog()
        self.LE_s1_tilt.setText(f"{self.CF.tilt:.3f}")
        self.CF.rlRF = proc_base.tilt_correction(self.CF.logRF, self.CF.tilt, cubic=True)

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
        self.s2_make()
    
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
        
        for ax in self.ax_sub[0]:
            ax.cla()

        self.im_s3_1 = self.ax_sub[0][0].imshow(self.CF.rmFlat[self.frameNum,5:-5,5:-5], plt.cm.gray, origin='lower')
        self.ax_sub[0][0].set_xlabel('Wavelength (pix)')
        self.ax_sub[0][0].set_ylabel('Slit (pix)')
        self.ax_sub[0][0].set_title('Atlas Subtraction')
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
        data = np.abs(self.wvlet_y[self.frameNum].wavelet.mean((0,2)))
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
        self.yFringe = [None]*self.CF.nf
        self.s1 = [None]*self.CF.nf
        self.ms1 = [None]*self.CF.nf
        for i in range(self.CF.nf):
            self.yFringe[i] = proc_base.cal_fringeSimple(self.wvlet_y[i], [self.yf_min, self.yf_max]).T
            self.s1[i] = self.CF.rmFlat[i] - self.yFringe[i]
            self.ms1[i] = self.CF.rmFlat[i] - self.yFringe[i]
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
            tt = self.s1[i][5:-5,5:-5].mean(0) - self.s1[i][5:-5,5:-5].min()
            whmin = self.cpos[i]
            pars = [tt[whmin-5], whmin, 5]
            cp, cr = curve_fit(proc_base.Gaussian, x[whmin-5:whmin+5], tt[whmin-5-5:whmin-5+5], p0=pars)
            # self.cpos[i] = int(cp[1])
            w[i] = int(cp[2]*1.5)

        self.msk_width = int(np.median(w))
        self.LE_s3_3_MW.setText(f"{self.msk_width}")
        self.ms1 = [None]*self.CF.nf        
        for i in range(self.CF.nf):
            self.ms1[i] = proc_base.data_mask_and_fill(self.s1[i], [self.cpos[i]-self.msk_width, self.cpos[i]+self.msk_width])

        self.mskShow = True
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
        
        self.fig.canvas.draw_idle()

    def s3_3_Apply(self):
        self.log += f"> Change the width of the mask.<br>"
        self._writeLog()
        self.msk_wdith = int(self.LE_s3_3_MW.text())

        for i in range(self.CF.nf):
            self.ms1[i] = proc_base.data_mask_and_fill(self.s1[i], [self.cpos[i]-self.msk_width, self.cpos[i]+self.msk_width])

        self.p_s3_3_mskMin.set_xdata([self.cpos[self.frameNum]-5-self.msk_width, self.cpos[self.frameNum]-5-self.msk_width])
        self.p_s3_3_mskMax.set_xdata([self.cpos[self.frameNum]-5+self.msk_width, self.cpos[self.frameNum]-5+self.msk_width])

        if self.mskShow == True:
            self.im_s3_3.set_data(self.ms1[self.frameNum][5:-5,5:-5])

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

        for i in range(self.CF.nf):
            self.wvlet_x[i] = Wavelet(self.ms1[i][5:-5,5:-5], dt=1, axis=1, dj=0.05, param=12)

        self.log += "> Done.<br>"
        self._writeLog()
        self.s3_4_wvShow()
        self.B_s3_4_wvShow.setEnabled(True)
        self.B_s3_4_FRapply.setEnabled(True)
        self.B_s3_4_simple.setEnabled(True)
        self.B_s3_4_gauss.setEnabled(True)

    def s3_4_wvShow(self):
        self.xwvShow = True
        self.XFshow = False
        self.s2Show = False
        self.ax_sub[3][0].set_visible(True)
        self.ax_sub[3][1].set_visible(False)
        nfreq = self.wvlet_x[self.frameNum].wavelet.shape[1]
        self.ax_sub[3][0].cla()
        data = np.abs(self.wvlet_x[self.frameNum].wavelet.mean((0,2)))
        xwh = data[:nfreq-20].argmax()
        self.xf_min = xwh-4
        self.xf_max = xwh+4
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
        self.xFringe = [None]*self.CF.nf
        self.s2 = np.zeros((self.CF.nf, self.CF.ny, self.CF.nw))
        for i in range(self.CF.nf):
            self.xFringe[i] = proc_base.cal_fringeSimple(self.wvlet_x[i], [self.xf_min, self.xf_max])
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
        self.imFx.set_data(self.xFringe[self.frameNum])
        self.imFx.set_clim(self.xFringe[self.frameNum].min(),self.xFringe[self.frameNum].max())
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
            data = np.abs(self.wvlet_x[self.frameNum].wavelet.mean((0,2)))
            self.pwvx.set_ydata(data)
        else:
            data = self.CF.rmFlat[self.frameNum][5:-5,5:-5]
            self.imFx.set_data(data)

        
        self.fig.canvas.draw_idle()