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
import fisspy.io.data

def alignoffset(image,template,cor=False):
    nxi,nyi=image.shape
    nxt,nyt=template.shape
    
    if nxi==nxt and nyi==nyt:
        raise ValueError('Two images are incompatible')
    
    