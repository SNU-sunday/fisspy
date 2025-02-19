from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from ..read import FISS
from ._mlsi_base import Model, RadLoss
from ..correction import corAll
from time import time
from os.path import join, dirname, basename, isdir
from os import mkdir

def MLSI4file(ifile, ofile=None, logfile=None, subsec=None, ncore=-1, quiet=True):
    """
    """
    ts = time()
    a = FISS(ifile)
    pa2 = corAll(a, subsec)
    nc = cpu_count()

    ncc = np.minimum(nc, ncore)
    if ncc == -1:
        ncc = nc

    if not quiet:
        t1 = time()
        dummy = Model(a.Rwave, pa2[0,0], ncore=1)
        t2 = time()
        x1, x2, y1, y2 = subsec
        expT = (t2-t1)*((x2-x1)*(y2-y1))/ncc
        print(f"It will take about {expT:.1f}+-{expT*0.2:.1f} seconds.")
    
    p, i0, i1, i2, epsD, epsP = Model(a.Rwave, pa2, line=a.line, ncore=ncc)
    RL1, RL2 = RadLoss(p, line='ha')

    sdir = dirname(ifile)
    sdir = join(sdir, 'inv')
    if ofile is None:
        bn = basename(ifile)
        of = bn.replace('c.fts','par.fts')
        sname = join(sdir, of)
    else:
        if not dirname(ofile):
            sname = join(sdir, ofile)
        sname = ofile
    print(sname)

    te = time()
    if not quiet:
        print(f"MLSI4file-Runtime: {te-ts:.0f} seconds.")
    


class MLSI:
    def __init__(self):
        None
    def TLI(self):
        None