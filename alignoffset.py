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
    
    if ndim>3 or ndim==1:
        raise ValueError('Image must be 2 or 3 dimensional array.')
    
    if not st[-1]==si[-1] and st[-2]==si[-2]:
        raise ValueError('Image and template are incompatible\n'
        'The shape of image = %s\n The shape of template = %s.'
        %(repr(si[-2:]),repr(st)))
    
    nx=st[-1]
    ny=st[-2]
    
    image=(image.T-image.mean(axis=(-1,-2))).T
    template-=template.mean()
    
    sigx=nx/6.
    sigy=ny/6.
    gx=np.arange(-nx/2,nx/2,1)
    gy=np.arange(-ny/2,ny/2,1)[:,np.newaxis]    
    gauss=np.exp(-0.5*((gx/sigx)**2+(gy/sigy)**2))**0.5
    
    #give the cross-correlation weight on the image center
    #to avoid the fast change the image by the granular motion or strong flow
    
    cor=ifft2(ifft2(image*gauss)*fft2(template*gauss)).real
    
    # calculate the cross-correlation values by using convolution theorem and 
    # DFT-IDFT relation
    
    s=np.where((cor.T==cor.max(axis=(-1,-2))).T)
    x0=s[-1]-nx*(s[-1]>nx/2)
    y0=s[-2]-ny*(s[-2]>ny/2)
    
    if ndim==2:
        cc=cor[s[0]-1:s[0]+2,s[1]-1:s[1]+2]
    else:
        cc=np.empty((si[0],3,3))
        cc[:,0,1]=cor[s[0],s[1]-1,s[2]]
        cc[:,1,0]=cor[s[0],s[1],s[2]-1]
        cc[:,1,1]=cor[s[0],s[1],s[2]]
        cc[:,1,2]=cor[s[0],s[1],s[2]+1]
        cc[:,2,1]=cor[s[0],s[1]+1,s[2]]
    
    x1=0.5*(cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-2.*cc[1,1])
    y1=0.5*(cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-2.*cc[1,1])
    
    x=x0+x1
    y=y0+y1
    
    return x, y

