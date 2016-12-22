from __future__ import absolute_import, print_function, division

from sunpy.image.rescale import resample as rescale
from scipy.ndimage.interpolation import shift as sci_shift
import numpy as np
from interpolation.splines import LinearSpline

def rot_trans(x,y,xc,yc,angle,dx=0,dy=0,inv=False):
    """
    rot_trans
    
    rotational transpose for input array of x, y and angle.
    
    
    """
    
    if not inv:
        xt=(x-xc)*np.cos(angle)+(y-yc)*np.sin(angle)+xc+dx
        yt=-(x-xc)*np.sin(angle)+(y-yc)*np.cos(angle)+yc+dy
    else:
        xt=(x-xc-dx)*np.cos(angle)-(y-yc-dy)*np.sin(angle)+xc
        yt=(x-xc-dx)*np.sin(angle)+(y-yc-dy)*np.cos(angle)+yc
    return xt,yt

def img_interpol(img,xa,ya,xt,yt,missing=-1):
    """
    img_interpol
    """
    shape=xt.shape
    size=xt.size
    smin=[ya[0,0],xa[0]]
    smax=[ya[-1,0],xa[-1]]
    order=[len(ya),len(xa)]
    interp=LinearSpline(smin,smax,order,img)
    
    a=np.array((yt.reshape(size),xt.reshape(size)))
    b=interp(a.T)
    res=b.reshape(shape)
    if missing!=-1:
        mask=np.invert((xt<xa.max())*(xt>xa.min())*(yt<ya.max())*(yt>ya.min()))
        res[mask]=missing
    return res

def rotation(img,angle,x,y,xc,yc,dx=0,dy=0,inv=False,missing=-1):
    xt,yt=rot_trans(x,y,xc,yc,angle,dx,dy,inv)
    return img_interpol(img,x,y,xt,yt,missing=missing)
    
def rot(img,angle,xc=False,yc=False,dx=0,dy=0,xmargin=0,ymargin=0,missing=0):
    ny,nx=img.shape
    nx+=xmargin
    ny+=ymargin
    xa=np.arange(nx)
    ya=np.arange(ny)[:,None]
    if not xc:
        xc=nx/2
    if not yc:
        yc=ny/2
    return rotation(img,angle,xa,ya,xc,yc,dx,dy,missing=missing)
    
def shift(image,x,y):
    """
    Imageshift
    
    Shifting the input image by x and y.
    The x and y values are the outputs of the alignoffset.
    
    Parameter
    Image : A 2 Dimensional array.
    x, y  : The align offset values for image.
            These are the outputs of the alignoffset code.
    ==============================
    Example)
    >>> newimage=imageshift(image,x,y)
    """
    
    return sci_shift(image,[y,x])