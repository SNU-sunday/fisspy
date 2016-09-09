"""
Align

Based on the alignoffset.pro IDL code written by Chae 2004

Parameters
Image :
Template :

Keyword
Cor : If True, then return values are x, y offsets
      and 2-Dimensional correlation value array of the two input images
      Default is False.

"""
from __future__ import absolute_import, print_function, division

__author__="J. Kang : jhkang@astro.snu.ac.kr"
__date__="Sep 01 2016"

from scipy.fftpack import ifft2,fft2
import numpy as np

def alignoffset(image,template,cor=False):
    """"""
    st=template.shape
    si=image.shape
    ndim=image.ndim
    
    if not st[-1]==si[-1] and st[-2]==si[-2]:
        raise ValueError('image and template are incompatible\n'
        'The shape of image = %s\n The shape of template = %s'
        %(repr(si[-2:]),repr(st)))
    
    nx=st[-1]
    ny=st[-2]
    
    try:
        image.T-=image.mean(axis=(-1,-2))
    template-=template.mean()
    
    sigx=nx/6.
    sigy=ny/6.
    gx=np.arange(-nx/2,nx/2,1)
    gy=np.arange(-ny/2,ny/2,1)[:,np.newaxis]    
    gauss=np.exp(-0.5*((gx/sigx)**2+(gy/sigy)**2))**0.5
    
    
    
    
    
    
    