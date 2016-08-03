"""
Read one frame from a FISS file or its PCA file.

Based on the idl code FISS_READ_FRAME.pro written by (J. Chae May 2013)
"""
from __future__ import absolute_import, division, print_function


__date__="Jul 29 2016"
__author__="J. Kang : jhkang@astro.snu.ac.kr

import astropy.io import fits



def frame(file,x1,x2=False,pca=False,ncoeff=False):
    if not file:
        raise ValueError('Empty filename: %s' % repr(name))
    
    header=fits.getheader(file)
    pfile=bool(header['pfile'])
    
    if not x2:
        x2=x1
    
    if pfile or pca:
        spec=pca_read(file,x,header,ncoeff=ncoeff)
    else:
        spec=fits.getdata(file)