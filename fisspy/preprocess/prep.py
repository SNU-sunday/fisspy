from fisspy.preprocess.proc_base import *
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from os.path import isdir, join, dirname, basename
from os import getcwd, makedirs
from glob import glob

class runPrep:
    def __init__(self, basedir, focdir=None):
        self.rcaldir = join(basedir, 'cal')
        self.pcaldir = join(basedir, 'proc', 'cal')
        self.procdir = join(basedir, 'proc')
        self.compdir = join(basedir, 'comp')
        self.rawdir = join(basedir, 'raw')

        lcam = ['A', 'B']
