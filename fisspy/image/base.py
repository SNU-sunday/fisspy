"""
Basic image process tool.
"""

from __future__ import absolute_import, division

import numpy as np
from interpolation.splines import LinearSpline, CubicSpline
from scipy.fftpack import ifft2, fft2
# from sunpy.coordinates import frames
# from astropy.coordinates import SkyCoord
# from sunpy.coordinates.ephemeris import get_earth
# from sunpy.physics.differential_rotation import solar_rotate_coordinate

__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ['alignoffset', 'rot_trans', 'img_interpol',
           'rotation', 'rot', 'shift']

def alignoffset(image0, template0, cor= None, test=False):
    """
    Calculate the align offset between two two-dimensional images

    Parameters
    ----------
    image0 : `~numpy.ndarray`
        Images for coalignment with the template
        2 Dimensional array
    template0 : `~numpy.ndarray`
        The reference image for coalignment
        2-Dimensional arry ex) template[y,x]
    cor: `bool`
        If True, return the correlation between template0 and result.

    Returns
    -------
    sh : `~numpy.ndarray`
        Shifted value of the image0
        np.array([yshift, xshift])

    Notes
    -----
        This code is based on the IDL code ALIGNOFFSET.PRO
        written by J. Chae 2004.
        Using for loop is faster than inputing the 3D array as,
            >>> res=np.array([alignoffset(image[i],template) for i in range(nt)])
        where nt is the number of elements for the first axis.

    Example
    -------
    >>> sh = alignoffset(image,template)
    """
    st=template0.shape
    si=image0.shape
    ndim=image0.ndim

    if ndim>=3 or ndim==1:
        raise ValueError('Image must be 2 or 3 dimensional array.')

    if not st[-1]==si[-1] and st[-2]==si[-2]:
        raise ValueError('Image and template are incompatible\n'
        'The shape of image = %s\n The shape of template = %s.'
        %(repr(si[-2:]),repr(st)))

    if not ('float' in str(image0.dtype) and 'float' in str(template0.dtype)):
        image0=image0.astype(float)
        template0=template0.astype(float)

    nx=st[-1]
    ny=st[-2]

    template=template0.copy()
    image=image0.copy()

    image=(image.T-image.mean(axis=(-1,-2))).T
    template-=template.mean()

    sigx=nx/6.
    sigy=ny/6.
    gx=np.arange(-nx/2,nx/2,1)
    gy=np.arange(-ny/2,ny/2,1)[:,np.newaxis]
    gauss=np.exp(-0.5*((gx/sigx)**2+(gy/sigy)**2))**0.5

    #give the cross-correlation weight on the image center
    #to avoid the fast change the image by the granular motion or strong flow

    corr=ifft2(ifft2(template*gauss)*fft2(image*gauss)).real
    # calculate the cross-correlation values by using convolution theorem and
    # DFT-IDFT relation

    s=np.where((corr.T==corr.max(axis=(-1,-2))).T)
    x0=s[-1]-nx*(s[-1]>nx/2)
    y0=s[-2]-ny*(s[-2]>ny/2)

    cc=np.empty((3,3))
    cc[0,1]=corr[s[0]-1,s[1]]
    cc[1,0]=corr[s[0],s[1]-1]
    cc[1,1]=corr[s[0],s[1]]
    cc[1,2]=corr[s[0],s[1]+1-nx]
    cc[2,1]=corr[s[0]+1-ny,s[1]]
    x1=0.5*(cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-2.*cc[1,1])
    y1=0.5*(cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-2.*cc[1,1])


    x=x0+x1
    y=y0+y1
    if test:
        return corr
    if cor:
        img = shift(image, [-y, -x])
        xx = np.arange(nx) + x
        yy = np.arange(ny) + y
        kx = np.logical_and(xx >= 0, xx <= nx - 1)
        ky = np.logical_and(yy >= 0, yy <= ny - 1)
        roi = np.logical_and(kx, ky[:,None])
        cor = (img*template)[roi].sum()/np.sqrt((img[roi]**2).sum() *
                      (template[roi]**2).sum())
        return np.array([y, x]), cor
    else:
        return np.array([y, x])

def rot_trans(x, y, xc, yc, angle, dx=0, dy=0, inv=False):
    """
    Rotational transpose for input array of x, y and angle.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Row vector of x.
    y : `~numpy.ndarray`
        Colomn vector of y.
    xc : `float`
        x-axis value of roatation center.
    yc : `float`
        y-axis value of rotation center.
    angle : `float`
        Roation angle in 'radian' unit.
    dx : (optional) `float`
        The relative displacement along x-axis
        of the rotated images to the reference image.
    dy : (optional) `float`
        The relative displacement along y-axis
        of the rotated images to the reference image.
    inv : (optional) `bool`
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

def img_interpol(img, xa, ya, xt, yt, missing=-1, cubic=False):
    """
    Interpolate the image for a given coordinates.

    Parameters
    ----------
    img : `~numpy.ndarray`
        N-dimensional array of image.
    xa : `~numpy.ndarray`
        Row vector of x.
    ya : `~numpy.ndarray`
        Colomn vector of y.
    xt : `~numpy.ndarray`
        Coordinates of the positions in the observed frame.
    yt : `~numpy.ndarray`
        Coordinates of the positions in the observed frame.
    missing : (optional) `float`
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.

    Returns
    -------
    res : ~numpy.ndarray
        N-dimensional interpolated image.
        The size of res is same as input img.

    """
    shape = img.shape
    ndim = img.ndim
    size = img.size
    ones = np.ones(shape)

    smin = np.zeros(ndim)
    smax = np.array(shape)-1
    smin[-1] = xa[0]
    smin[-2] = ya[0,0]
    smax[-1] = xa[-1]
    smax[-2] = ya[-1,0]
    order = shape
    if cubic:
        interp = CubicSpline(smin, smax, order, img)
    else:
        interp = LinearSpline(smin, smax, order, img)

    inp = np.zeros((ndim,size))
    for i, sh in enumerate(shape[:-2]):
        tmp = np.arange(sh)[tuple([None]*i + [Ellipsis] + [None]*(ndim-1-i))]*ones
        inp[i] = tmp.reshape(size)
        
    inp[-2] = (yt * ones).reshape(size)
    inp[-1] = (xt * ones).reshape(size)
    # a = np.array((yt.reshape(size),xt.reshape(size)))
    b = interp(inp.T)
    res=b.reshape(shape)
    if missing!=-1:
        mask=np.invert((xt<=xa.max())*(xt>=xa.min())*(yt<=ya.max())*(yt>=ya.min()))*ones.astype(bool)
        res[mask]=missing
    return res

def rotation(img, angle, x, y, xc, yc,
             dx=0, dy=0, inv=False, missing=-1, cubic=False):
    """
    Rotate the input image with angle and center position.

    Parameters
    ----------
    img : `~numpy.ndarray`
        N-dimensional array of image.
    x : `~numpy.ndarray`
        Row vector of x.
    y : `~numpy.ndarray`
        Colomn vector of y.
    xc : `float`
        x-axis value of roatation center.
    yc : `float`
        y-axis value of rotation center.
    angle : `float`
        Roation angle in 'radian' unit.
    dx : (optional) `float`
        The relative displacement along x-axis
        of the rotated images to the reference image.
    dy : (optional) `float`
        The relative displacement along y-axis
        of the rotated images to the reference image.
    inv : (optional) `bool`
        If True, the do inverse roattion transpose.
    missing : (optional) `float`
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.

    Returns
    -------
    result : `~numpy.ndarray`
        rotated image.

    Notes
    -----
    It is not conventional rotation.
    It is just used for the coalignment module.

    """
    xt,yt=rot_trans(x, y, xc, yc, angle,
                    dx, dy, inv)
    return img_interpol(img, x, y, xt, yt,
                        missing=missing, cubic=cubic)

def rot(img, angle, xc=False, yc=False,
        dx=0, dy=0, xmargin=0, ymargin=0, missing=0, cubic=False):
    """
    Rotate the input image.

    Parameters
    ----------
    img : `~numpy.ndarray`
        N-dimensional array of image.
    angle : `float`
        Roation angle in 'radian' unit.
    xc : (optional) `float`
        x-axis value of roatation center.
        Default is the image center.
    yc : (optional) `float`
        y-axis value of rotation center.
        Default is the image center.
    dx : (optional) `float`
        The relative displacement along x-axis
        of the rotated images to the reference image.
    dy : (optional) `float`
        The relative displacement along y-axis
        of the rotated images to the reference image.
    xmargin : (optional) `float`
        The margin value of x-axis
    ymargin : (optional) `float`
        The margin value of y-axis
    missing : (optional) `float`
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.

    Returns
    -------
    result : `~numpy.ndarray`
        rotated image.

    Notes
    -----
    The input angle must be in radian unit.

    """
    nx = img.shape[-1]
    ny = img.shape[-2]
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
    return img_interpol(img,x,y,xt,yt,missing=missing, cubic=cubic)

def shift(image, sh, missing=0, cubic=False):
    """
    Shift the given image.

    Parameters
    ----------
    image :  `~numpy.ndarray`
        2 dimensional array.
    sh : tuple, list or ndarray
        tuple, list or ndarray of shifting value set (y,x)
    missing: `float`
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.

    Returns
    -------
    simage : ~numpy.ndarray
        shifted image.
    """
    ny, nx = image.shape
    x=np.arange(nx)
    y=np.arange(ny)[:,None]
    xt=x-sh[1]+y*0
    yt=y-sh[0]+x*0

    return img_interpol(image,x,y,xt,yt,missing=missing,cubic=cubic)

def shift3d(img, sh):
    """
    Shift the given image.

    Parameters
    ----------
    image :  `~numpy.ndarray`
        3 dimensional array.
    sh : tuple, list or ndarray
        tuple, list or ndarray of shifting value set (y,x)

    Returns
    -------
    simage : ~numpy.ndarray
        shifted image.
    """
    nt, ny, nx =img.shape

    t = np.arange(nt)[:,None,None]
    y = np.arange(ny)[None,:,None]
    x = np.arange(nx)
    tt = t + y*0 + x*0
    yt = y - sh[0][:, None, None] + t*0 + x*0
    xt = x - sh[1][:, None, None] + t*0 + y*0

    return img_interpol3d(img, t, y, x, tt, yt, xt, missing=0)

def img_interpol3d(img, ta, ya, xa,
                   tt, yt, xt, missing=None):
    """
    Interpolate the image for a given coordinates.

    Parameters
    ----------
    img : `~numpy.ndarray`
        3 dimensional array of image.
    xa : `~numpy.ndarray`
        Row vector of x.
    ya : `~numpy.ndarray`
        Colomn vector of y.
    ta : `~numpy.ndarray`
        Frame vector.
    tt : `~numpy.ndarray`
        Coordinates of the positions in the observed frame.
    yt : `~numpy.ndarray`
        Coordinates of the positions in the observed frame.
    xt : `~numpy.ndarray`
        Coordinates of the positions in the observed frame.
    missing : (optional) `float`
        The value of extrapolated position.
        Default is -1, and it means the False.
        If False, then extrapolate the given position.

    Returns
    -------
    res : ~numpy.ndarray
        3 dimensional interpolated image.
        The size of res is same as input img.

    """
    shape = xt.shape
    size = xt.size
    smin = [ta[0,0,0], ya[0,0,0], xa[0]]
    smax = [ta[-1,0,0],ya[0,-1,0], xa[-1]]
    order = [ta.size, ya.size, xa.size]
    interp = LinearSpline(smin, smax, order, img)
    a = np.array((tt.reshape(size), yt.reshape(size), xt.reshape(size)))
    b=interp(a.T)
    res=b.reshape(shape)
    if missing is not None:
        mask=np.invert((xt<=xa.max())*(xt>=xa.min())*(yt<=ya.max())*(yt>=ya.min()))
        res[mask]=missing
    return res