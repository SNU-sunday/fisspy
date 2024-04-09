from .base import AlignOffset, rotImage, shiftImage
import numpy as np
import matplotlib.pyplot as plt
from ..read import FISS

def calAlignPar(lcam, refFrame=None):
    nf = len(lcam)
    if refFrame is None:
        refFrame = nf//2
    rf = refFrame
    fissr = FISS(lfiles[rf], wvCalibMethod='simple')
    imr = fissr.data[..., 50:55].mean(2)
    
def alignCams(imA, imB, refCam='A'):
    None

def alignRot(flist):
    None

def makeAlignCube(save=False):
    None

class Align:
    """
    """
    def __init__(self, lcamA, lcamB, refCam='A', refFrame=None, align=True):
        """
        Parameters
        ----------
        lcamA: `list`
            List of the camA files
        lcamB: `list`
            List of the camB files
        refCam: `str`
            Reference camera for the alignment.
            Either 'A' or 'B'.
            Default is 'A'.
        refFrame: `int`
            Reference frame number.
            Default is the middle frame of the input list.
        Returns
        -------
        """
        nfA = len(lcamA)
        nfB = len(lcamB)
        if nfA <= 1 or nfB <=1:
            raise ValueError(f"The number of elements of either lcamA or lcamB should be larger than 1.\n    Note) nfA={nfA}, nfB={nfB} .")
        if nfA != nfB:
            raise ValueError(f"lcamA and lcamB should have the same number of elements.")
        self.lcamA = lcamA
        self.lcamB = lcamB
        self.nf = nfA
        self.refCam = refCam.upper()
        if self.refCam != 'A' or self.refCam != 'B':
            raise ValueError("refCam should be either 'A' or 'B'.")
        
        if refFrame is None:
            refFrame = self.nf//2
        if refFrame < 0 or refFrame >= self.nf:
            raise ValueError("refFrame should be 0<=refFrame<nf, where nf is the number of files.")
        
        self.refFrame = refFrame


        
        