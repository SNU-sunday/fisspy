from PyQt5 import QtWidgets, QtCore, QtGui
from os.path import join, dirname, basename
import matplotlib.pyplot as plt
import numpy as np
from fisspy.preprocess import proc_base
from fisspy.analysis.wavelet import Wavelet
from glob import glob
from astropy.io import fits

class prepGUI:
    def __init__(self, basedir, ffocA=None, ffocB=None):
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
        
        self.fig = plt.figure(figsize=[17,7], num="FISS preprocess")

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
        self.ax_sub = [[ax_s3_1_1], [ax_s3_2_1, ax_s3_2_2]]


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
        
    def subStep(self, num):
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
        self.panel.setMaximumSize(QtCore.QSize(300, 4096))
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
        self.subStepWidgets = [None]*2

        # create Step0 Widget
        if True:
            self.L_s0_fflist = QtWidgets.QLabel()
            self.L_s0_fflist.setText("Flat file list")
            self.L_s0_fflist.setFont(self.fNormal)
            self.CB_s0_fflist = QtWidgets.QComboBox()
            self.CB_s0_fflist.setStyleSheet("background-color: %s; border: 1px solid %s;"%(self.bg_second, self.border))
            self.CB_s0_fflist.addItems(self.fflatGBL)
            self.CB_s0_fflist.setCurrentIndex(0)
            self.CB_s0_fflist.currentIndexChanged.connect(self.chFlat)
            self.vboxCtrl.addWidget(self.L_s0_fflist)
            self.vboxCtrl.addWidget(self.CB_s0_fflist)
            self.StepWidgets[0] = [self.L_s0_fflist, self.CB_s0_fflist]
            
        # create Step1 tilt Widget
        if True:
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

            self.vboxCtrl.addLayout(self.HL0_s1)
            self.vboxCtrl.addLayout(self.HL_s1)

            self.StepWidgets[1] = [self.L_s1_get_tilt, self.B_s1_run, self.L_s1_tilt, self.LE_s1_tilt, self.L_s1_deg, self.B_s1_Apply]

            self.B_s1_run.clicked.connect(self.s1_Run)
            self.B_s1_Apply.clicked.connect(self.s1_Apply)

        # create Step2 curvature Correction
        if True:
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

            self.vboxCtrl.addLayout(self.HL0_s2)
            self.HL_s2_coeff.addWidget(self.L_s2_p0)
            self.HL_s2_coeff.addWidget(self.LE_s2_p0)
            self.HL_s2_coeff.addWidget(self.L_s2_p1)
            self.HL_s2_coeff.addWidget(self.LE_s2_p1)
            self.HL_s2_coeff.addWidget(self.L_s2_p2)
            self.HL_s2_coeff.addWidget(self.LE_s2_p2)
            self.vboxCtrl.addLayout(self.HL_s2_coeff)
            self.vboxCtrl.addWidget(self.B_s2_Apply)

            self.StepWidgets[2] = [self.L_s2_get_curve, self.B_s2_run, self.L_s2_p0, self.LE_s2_p0, self.L_s2_p1, self.LE_s2_p1, self.L_s2_p2, self.LE_s2_p2, self.B_s2_Apply]

            self.B_s2_run.clicked.connect(self.s2_Run)
            self.B_s2_Apply.clicked.connect(self.s2_Apply)
            
        # create Step3-1
        if True:
            self.L_s3_1_subStep = QtWidgets.QLabel()
            self.L_s3_1_subStep.setText(self.List_subStep[0])
            self.L_s3_1_subStep.setFont(self.fNormal)
            self.L_s3_1_subStep.setStyleSheet(f"color: {self.font_third};")
            self.L_s3_1_subStep.setWordWrap(True)

            self.B_s3_1_run = QtWidgets.QPushButton()
            self.B_s3_1_run.setText("Run")
            self.B_s3_1_run.setFont(self.fNormal)
            self.B_s3_1_run.setStyleSheet(f"background-color: {self.bg_second};")

            self.vboxCtrl.addWidget(self.L_s3_1_subStep)
            self.vboxCtrl.addWidget(self.B_s3_1_run)

            self.subStepWidgets[0] = [self.L_s3_1_subStep, self.B_s3_1_run]

            self.B_s3_1_run.clicked.connect(self.s3_1_Run)

        # create Step3-2
        if True:
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

            self.yf_max = 120
            self.L_s3_2_FRmax = QtWidgets.QLabel()
            self.L_s3_2_FRmax.setText("max:")
            self.L_s3_2_FRmax.setFont(self.fNormal)
            self.LE_s3_2_FRmax = QtWidgets.QLineEdit()
            self.LE_s3_2_FRmax.setText("120")
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


            self.B_s3_2_FringeShow = QtWidgets.QPushButton()
            self.B_s3_2_FringeShow.setText("Show")
            self.B_s3_2_FringeShow.setFont(self.fNormal)
            self.B_s3_2_FringeShow.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_FringeShow.clicked.connect(self.s3_2_FringeShow)

            self.HL_Fringe.addWidget(self.B_s3_2_simple)
            self.HL_Fringe.addWidget(self.B_s3_2_FringeShow)

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

            self.B_s3_2_blink = QtWidgets.QPushButton()
            self.B_s3_2_blink.setText("Blink")
            self.B_s3_2_blink.setFont(self.fNormal)
            self.B_s3_2_blink.setStyleSheet(f"background-color: {self.bg_second};")
            self.B_s3_2_blink.clicked.connect(self.s3_2_blink)

            self.HL_s3_2_res.addWidget(self.B_s3_2_resShow)
            self.HL_s3_2_res.addWidget(self.B_s3_2_blink)

            self.vboxCtrl.addWidget(self.L_s3_2_subStep)
            self.vboxCtrl.addLayout(self.HL_wvlet)
            self.vboxCtrl.addWidget(self.L_s3_2_FR)
            self.vboxCtrl.addLayout(self.HL_FR)
            self.vboxCtrl.addWidget(self.L_s3_2_calFringe)
            self.vboxCtrl.addLayout(self.HL_Fringe)
            self.vboxCtrl.addWidget(self.L_s3_2_res)
            self.vboxCtrl.addLayout(self.HL_s3_2_res)

            self.subStepWidgets[1] = [self.L_s3_2_subStep, self.L_s3_2_wvlet, self.B_s3_wvCal, self.B_s3_wvShow, self.L_s3_2_FR, self.L_s3_2_FRmin, self.LE_s3_2_FRmin, self.L_s3_2_FRmax, self.LE_s3_2_FRmax, self.B_s3_2_FRapply, self.L_s3_2_calFringe, self.B_s3_2_simple, self.B_s3_2_FringeShow, self.L_s3_2_res, self.B_s3_2_resShow, self.B_s3_2_blink]
            self.imFy = None


        
        
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
            self.G_Log.setMinimumSize(QtCore.QSize(100, 300))
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

        # add layout
        self.vboxAll.addLayout(self.vboxMain)
        self.vboxAll.addLayout(self.vboxCtrl)
        self.vboxAll.addWidget(line2)
        self.vboxAll.addLayout(self.HL_PN)
        self.vboxAll.addItem(vSpacer)
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
        self.im = self.ax[0][0].imshow(self.CF.logRF[self.frameNum,5:-5,5:-5], plt.cm.gray, origin='lower', interpolation='nearest')
        self.ax[0][0].set_xlabel('Wavelength (pix)')
        self.ax[0][0].set_ylabel('Slit (pix)')
        self.ax[0][0].set_title(f'Raw Flat {self.CF.date}')
        self.fig.canvas.draw_idle()

        self.log += f"> Read Flat: {self.fflatGBL[self.fidx]}<br>"
        self._writeLog()

    def _writeLog(self):
        self.L_Log.setText(self.log)
        plt.pause(0.1)
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

        im = self.ax_sub[0][0].imshow(self.CF.rmFlat[self.frameNum,5:-5,5:-5], plt.cm.gray, origin='lower')
        self.ax_sub[0][0].set_xlabel('Wavelength (pix)')
        self.ax_sub[0][0].set_ylabel('Slit (pix)')
        self.ax_sub[0][0].set_title('Atlas Subtraction')
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
        self.log += "> Calculate Fringe Patterns.<br>"
        self._writeLog()
        self.yFringe = [None]*self.CF.nf
        self.s1 = [None]*self.CF.nf
        for i in range(self.CF.nf):
            self.yFringe[i] = proc_base.cal_fringeSimple(self.wvlet_y[i], [self.yf_min, self.yf_max]).T
            self.s1[i] = self.CF.rmFlat[i] - self.yFringe[i]
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

    def s3_2_FringeShow(self):
        self.ax_sub[1][0].set_visible(False)
        self.ax_sub[1][1].set_visible(True)
        self.ax_sub[1][1].set_title('Fringe Pattern (y-dir)')
        self.imFy.set_data(self.yFringe[self.frameNum][5:-5,5:-5])
        self.imFy.set_clim(self.yFringe[self.frameNum][5:-5,5:-5].min(),self.yFringe[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()

    def s3_2_resShow(self):
        self.s1Show = True        
        self.ax_sub[1][0].set_visible(False)
        self.ax_sub[1][1].set_visible(True)
        self.ax_sub[1][1].set_title('y-dir Fringe Subtracted')
        self.imFy.set_data(self.s1[self.frameNum][5:-5,5:-5])
        self.imFy.set_clim(self.s1[self.frameNum][5:-5,5:-5].min(),self.s1[self.frameNum][5:-5,5:-5].max())
        self.fig.canvas.draw_idle()

    def s3_2_blink(self):
        self.s1Show = not self.s1Show
        if self.s1Show:
            self.ax_sub[1][1].set_title('y-dir Fringe Subtracted')
            self.imFy.set_data(self.s1[self.frameNum][5:-5,5:-5])
        else:
            self.ax_sub[1][1].set_title('Original')
            self.imFy.set_data(self.CF.rmFlat[3][5:-5,5:-5])
        self.fig.canvas.draw_idle()