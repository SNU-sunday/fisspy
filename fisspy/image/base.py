"""
Basic image process tool.
"""

from __future__ import absolute_import, division

from sunpy.image.rescale import resample as rescale
import numpy as np
from interpolation.splines import LinearSpline

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ['rescale', 'rot_trans', 'img_interpol',
           'rotation', 'rot', 'shift']

def rot_trans(x,y,xc,yc,angle,dx=0,dy=0,inv=False):
    """
    Rotational transpose for input array of x, y and angle.
    
    Parameters
    ----------
    x : ~numpy.ndarray
        Row vector of x.
    y : ~numpy.ndarray
        Colomn vector of y.
    xc : float
        x-axis value of roatation center.
    yc : float
        y-axis value of rotation center.
    angle : float
        Roation angle in 'radian' unit.
    dx : (optional) float
        The relative displacement along x-axis 
        of the rotated images to the reference image.
    dy : (optional) float
        The relative displacement along y-axis 
        of the rotated images to the reference image.
    inv : (optional) bool
        If True, the do inverse roattion transpose.
    
    Returns
    -------
    xt : ~numpy.ndarray
        Transposed coordinates of the positions in the observed frame
    yt : ~numpy.ndarray
        Transposed coordinates of the positions in the observed frame
        
    Notes
    -----
    The input angle must be in radian.
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
    Interpolate the image for a given coordinates.
    
    Parameters
    ----------
    img : ~numpy.ndarray
        2 dimensional array of image.
    x : ~numpy.ndarray
        Row vector of x.
    y : ~numpy.ndarray
        Colomn vector of y.
    xt : ~numpy.ndarray
        Coordinates of the positions in the observed frame.
    yt : ~numpy.ndarray
        Coordinates of the positions in the observed frame.
    missing : (optional) float
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.
    
    Returns
    -------
    res : ~numpy.ndarray
        2 dimensional interpolated image.
        The size of res is same as input img.
    
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
        mask=np.invert((xt<=xa.max())*(xt>=xa.min())*(yt<=ya.max())*(yt>=ya.min()))
        res[mask]=missing
    return res

def rotation(img,angle,x,y,xc,yc,dx=0,dy=0,inv=False,missing=-1):
    """
    Rotate the input image with angle and center position.
    
    Parameters
    ----------
    img : ~numpy.ndarray
        2 dimensional array of image.
    x : ~numpy.ndarray
        Row vector of x.
    y : ~numpy.ndarray
        Colomn vector of y.
    xc : float
        x-axis value of roatation center.
    yc : float
        y-axis value of rotation center.
    angle : float
        Roation angle in 'radian' unit.
    dx : (optional) float
        The relative displacement along x-axis 
        of the rotated images to the reference image.
    dy : (optional) float
        The relative displacement along y-axis 
        of the rotated images to the reference image.
    inv : (optional) bool
        If True, the do inverse roattion transpose.
    missing : (optional) float
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.
    
    Returns
    -------
    result : ~numpy.ndarray
        rotated image.
        
    Notes
    -----
    It is not conventional rotation.
    It is just used for the coalignment module.
    
    """
    xt,yt=rot_trans(x,y,xc,yc,angle,dx,dy,inv)
    return img_interpol(img,x,y,xt,yt,missing=missing)
    
def rot(img,angle,xc=False,yc=False,dx=0,dy=0,xmargin=0,ymargin=0,missing=0):
    """
    Rotate the input image.
    
    Parameters
    ----------
    img : ~numpy.ndarray
        2 dimensional array of image.
    angle : float
        Roation angle in 'radian' unit.
    xc : (optional) float
        x-axis value of roatation center.
        Default is the image center.
    yc : (optional) float
        y-axis value of rotation center.
        Default is the image center.
    dx : (optional) float
        The relative displacement along x-axis 
        of the rotated images to the reference image.
    dy : (optional) float
        The relative displacement along y-axis 
        of the rotated images to the reference image.
    xmargin : (optional) float
        The margin value of x-axis
    ymargin : (optional) float
        The margin value of y-axis
    missing : (optional) float
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.
    
    Returns
    -------
    result : ~numpy.ndarray
        rotated image.
    
    Notes
    -----
    The input angle must be in radian unit.
    
    """
    ny,nx=img.shape
    nx1=int(nx+2*xmargin)
    ny1=int(ny+2*ymargin)
    x=np.arange(nx)
    y=np.arange(ny)[:,None]
    xa=np.arange(nx1)-xmargin
    ya=(np.arange(ny1)-ymargin)[:,None]
    

    if not xc:
        xc=nx/2
    if not yc:
        yc=ny/2
    xt, yt=rot_trans(xa,ya,xc,yc,angle,dx=dx,dy=dy)
    return img_interpol(img,x,y,xt,yt,missing=missing)
    
def shift(image,shift):
    """
    Shift the given image.
    
    Parameters
    ----------
    image :  ~numpy.ndarray
        2 dimensional array.
    shift : tuple, list or ndarray
        tuple, list or ndarray of shifting value set (y,x)
    
    Returns
    -------
    simage : ~numpy.ndarray
        shifted image.
    """
    ny, nx =image.shape
    x=np.arange(nx)
    y=np.arange(ny)[:,None]
    xt=x-shift[1]+y*0
    yt=y-shift[0]+x*0
    
    return img_interpol(image,x,y,xt,yt,missing=0)