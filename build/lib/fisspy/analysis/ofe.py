from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve as conv
from os import getcwd, mkdir, extsep
from os.path import join, basename, dirname, isdir
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

__all__ = ["runDAVE", "readOFE"]

def runDAVE(data0, output=False, overwrite=False, fwhm=10, adv=1, source=0,
            noise=1, winFunc='Gaussian', outSig=False):
    """
    Differential Affine Velocity Estimator for all spatial points.
    This is the python function of dave_multi.pro IDL code written by J. Chae (2009).

    Parameters
    ----------
    data0: `~numpy.ndarray` or fits file.
        Three-dimensional input array with shape (nt, ny, nx) or
        fits file with same dimension and shape.
    output: `str`, optional
        The name of the output fits file name.
        If False, it makes OFE directory and write the *_dave.fits file.
        Default is `False`.
    fwhm: `int`, optional
        FWHM of the window function (should be even positive integer)

    Returns
    -------
    fits file:
    """
    if type(data0) == str:
        data = fits.getdata(data0)
        dirn = join(dirname(data0), 'ofe')
        if not isdir(dirn):
            mkdir(dirn)
        fname = f'{basename(data0).split(extsep)[0]}_dave.fits'
    else:
        data = data0
        dirn = getcwd()
        fname = 'dave.fits'
    if data.ndim != 3:
        raise ValueError('data must be 3-D array.')
    if not output:
        output = join(dirn, fname)

    dnt, dny, dnx = data.shape
    psw = adv
    qsw = source

    # Construncting window function
    winFunc = winFunc.capitalize()
    h = int(fwhm/2)
    if winFunc == 'Square':
        mf = 1
    else:
        mf = 2
    nx = 2*h*mf+1
    ny = 2*h*mf+1
    x = -(np.arange(nx)-nx//2)
    y = -(np.arange(ny)-ny//2)

    if winFunc == 'Square':
        w = np.ones((ny, nx))
    elif winFunc == 'Gaussian':
        w = np.exp(-np.log(2)*((x/h)**2+(y[:,None]/h)**2))
    elif winFunc == 'Hanning':
        w = (1+np.cos(np.pi*x/h/2))*(1+np.cos(np.pi*y/h/2))/4
    else:
        raise ValueError("winFunc must be one of ('Square', 'Gaussian', "
                         "'Hanning')")
    w /= noise**2

    # Construncting coefficent arrays
    im = data
    imT = (np.roll(im, -1, axis=0) - np.roll(im, 1, axis=0))/2
    imY, imX = np.gradient(im, axis=(1, 2))

    npar = 6+qsw
    A = np.empty((npar, npar, dnt, dny, dnx))
    A[0,0] = conv(imX*imX, w[None, :, :],
                  'same', axes=(1, 2))                          # U0, U0
    A[1,0] = A[0,1] = conv(imY*imX, w[None, :, :],
                           'same', axes=(1, 2))                 # V0, U0
    A[1,1] = conv(imY*imY, w[None, :, :],
                  'same', axes=(1, 2))                          # V0, V0
    A[2,0] = A[0,2] = conv(imX*imX, x*w[None, :, :],
                           'same', axes=(1, 2)) \
                    + psw*conv(imX*im, w[None, :, :],
                               'same',axes=(1, 2))              # Ux, U0
    A[2,1] = A[1,2] = conv(imX*imY, x*w[None, :, :],
                           'same', axes=(1, 2)) \
                    + psw*conv(imY*im, w[None, :, :],
                               'same', axes=(1, 2))             # Ux, V0
    A[2,2] = conv(imX*imX, x*x*w[None, :, :],
                  'same', axes=(1, 2)) \
           + 2*psw*conv(imX*im, x*w[None, :, :],
                        'same', axes=(1, 2)) \
           +  psw**2*conv(im*im, w[None, :, :],
                          'same', axes=(1, 2))                  # Ux, Ux
    A[3,0] = A[0,3] = conv(imY*imX, y[None,:,None]*w[None, :, :],
                           'same', axes=(1, 2)) \
                    + psw*conv(imX*im, w[None, :, :],
                               'same', axes=(1, 2))             # Vy, U0
    A[3,1] = A[1,3] = conv(imY*imY, y[None,:,None]*w[None, :, :],
                           'same', axes=(1, 2)) \
                    + psw*conv(imY*im, w[None, :, :],
                               'same', axes=(1, 2))             # Vy, V0
    A[3,2] = A[2,3] = conv(imY*imX, y[None,:,None]*x*w[None, :, :],
                           'same', axes=(1, 2)) \
                    + psw*conv(imY*im, y[None,:,None]*w[None, :, :],
                               'same', axes=(1, 2)) + \
                    + psw*conv(imX*im, x*w[None, :, :],
                               'same', axes=(1, 2)) \
                    + psw**2*conv(im*im, w[None, :, :],
                                  'same', axes=(1, 2))          # Vy, Ux
    A[3,3] = conv(imY*imY, y[None,:,None]*y[None,:,None]*w[None, :, :],
                  'same', axes=(1, 2)) \
           + 2*psw*conv(imY*im, y[None,:,None]*w[None, :, :],
                        'same', axes=(1, 2)) \
           + psw**2*conv(im*im, w[None, :, :],
                         'same', axes=(1, 2))                   # Vy, Vy
    A[4,0] = A[0,4] = conv(imX*imX, y[None,:,None]*w[None, :, :],
                           'same', axes=(1, 2))                 # Uy, U0
    A[4,1] = A[1,4] = conv(imX*imY, y[None,:,None]*w[None, :, :],
                           'same', axes=(1, 2))                 # Uy, V0
    A[4,2] = A[2,4] = conv(imX*imX, y[None,:,None]*x*w[None, :, :],
                           'same', axes=(1, 2)) \
                    + psw*conv(imX*im, y[None,:,None]*w[None, :, :],
                               'same', axes=(1, 2))             # Uy, Ux
    A[4,3] = A[3,4] = conv(imX*imY,
                           y[None,:,None]*y[None,:,None]*w[None,:,:],
                           'same', axes=(1, 2)) \
                    + psw*conv(imX*im, y[None,:,None]*w[None,:,:],
                               'same', axes=(1, 2))             # Uy, Vy
    A[4,4] = conv(imX*imX, y[None,:,None]*y[None,:,None]*w[None,:,:],
                  'same', axes=(1, 2))                          # Uy, Uy
    A[5,0] = A[0,5] = conv(imY*imX, x*w[None,:,:],
                           'same', axes=(1, 2))                 # Vx, U0
    A[5,1] = A[1,5] = conv(imY*imY, x*w[None,:,:],
                           'same', axes=(1, 2))                 # Vx, V0
    A[5,2] = A[2,5] = conv(imY*imX, x*x*w[None,:,:],
                           'same', axes=(1, 2)) \
                    + psw*conv(im*imY, x*w[None,:,:],
                               'same', axes=(1, 2))             # Vx, Ux
    A[5,3] = A[3,5] = conv(imY*imY, x*y[None,:,None]*w[None,:,:],
                           'same', axes=(1, 2)) \
                    + psw*conv(im*imY, x*w[None,:,:],
                               'same', axes=(1, 2))             # Vx, Vy
    A[5,4] = A[4,5] = conv(imY*imX, x*y[None,:,None]*w[None,:,:],
                           'same', axes=(1, 2))                 # Vx, Uy
    A[5,5] = conv(imY*imY, x*x*w[None,:,:],
                  'same', axes=(1, 2))                          #Vx, Vx

    if qsw:
        A[6,0] = A[0,6] = -qsw*conv(im*imX, w[None,:,:],
                                    'same', axes=(1, 2))        # mu, U0
        A[6,1] = A[1,6] = -qsw*conv(im*imY, w[None,:,:],
                                    'same', axes=(1, 2))        # mu, V0
        A[6,2] = A[2,6] = -qsw*conv(im*imX, x*w[None,:,:],
                                    'same', axes=(1, 2)) \
                        - qsw*psw*conv(im*im, w[None,:,:],
                                       'same', axes=(1, 2))     # mu, Ux
        A[6,3] = A[3,6] = -qsw*conv(im*imY, y[None,:,None]*w[None,:,:],
                                    'same', axes=(1, 2)) \
                        - qsw*psw*conv(im*im, w[None,:,:],
                                       'same', axes=(1, 2))     # mu, Vy
        A[6,4] = A[4,6] = -qsw*conv(im*imX, y[None,:,None]*w[None,:,:],
                                    'same', axes=(1,2))         # mu, Uy
        A[6,5] = A[5,6] = -qsw*conv(im*imY, x*w[None,:,:],
                                    'same', axes=(1, 2))        # mu, Vx
        A[6,6] = -qsw**2*conv(im*im, w[None,:,:],
                              'same', axes=(1,2))               # mu, mu

    B = np.empty((npar, dnt, dny, dnx))
    B[0] = conv(imT*imX, -w[None,:,:], 'same', axes=(1,2))
    B[1] = conv(imT*imY, -w[None,:,:], 'same', axes=(1,2))
    B[2] = conv(imT*imX, -x*w[None,:,:], 'same', axes=(1,2)) \
         + psw*conv(imT*im, -w[None,:,:], 'same', axes=(1,2))
    B[3] = conv(imT*imY, -y[None,:,None]*w[None,:,:], 'same', axes=(1,2)) \
         + psw*conv(imT*im, -w[None,:,:], 'same', axes=(1,2))
    B[4] = conv(imT*imX, -y[None,:,None]*w[None,:,:], 'same', axes=(1,2))
    B[5] = conv(imT*imY, -x*w[None,:,:], 'same', axes=(1,2))
    if qsw:
        B[6] = qsw*conv(imT*(-im), -w[None,:,:], 'same', axes=(1,2))

    dave = np.linalg.solve(A.T, B.T).T


    if not outSig:
        hdu = fits.PrimaryHDU(dave)
        hdu.header['type'] = 'DAVE'
        hdu.writeto(output, overwrite=overwrite)
#    else: #TODO sigma and chisq calculation
    return output


class readOFE:
    def __init__(self, data0, ofeFile, scale=None, dt=None, gMethod=True):
        """
        Read the Optical Flow Estimated File.

        Parameters
        ----------
        scale: float
            Spatial pixel scale (arcsec).
        """

        if type(data0) == str:
            data = fits.getdata(data0)
            header = fits.getheader(data0)
            nx = header['naxis1']
            ny = header['naxis2']
            cx = header['crval1']
            cy = header['crval2']
            cxp = header['crpix1']
            cyp = header['crpix2']
            dx = header['cdelt1']
            dy = header['cdelt2']
            dt = header['cdelt3']
            l = -(cxp+0.5)*dx+cx
            b = -(cyp+0.5)*dy+cy
            r = (nx-(cxp+0.5))*dx+cx
            t = (ny-(cyp+0.5))*dy+cy

            scale = dx
        else:
            data = data0.copy()
            l = -0.5
            b = -0.5
            r = nx-0.5
            t = ny-0.5
            dx = 1
            dy = 1
        self.extent = [l, r, b, t]
        self.data = data
        self.nt, self.ny, self.nx = self.data.shape

        self._xarr = np.linspace(self.extent[0]+dx*0.5,
                                 self.extent[1]-dx*0.5,
                                 self.nx)
        self._yarr = np.linspace(self.extent[2]+dy*0.5,
                                 self.extent[3]-dy*0.5,
                                 self.ny)

        if self.data.ndim != 3:
            raise ValueError("data must have 3 dimension")
        if not scale or not dt:
            raise KeyError("If data is an `~numpy.ndarray`, "
                           "'scale' and 'dt' must be given.")
        unit = scale*725/dt # km/s
        self.ofe = fits.getdata(ofeFile)
        self.oheader = fits.getheader(ofeFile)
        self.otype = self.oheader['type']
        self.U0 = self.ofe[0]*unit
        self.V0 = self.ofe[1]*unit
        self.Ux = self.ofe[2]*unit
        self.Vy = self.ofe[3]*unit
        self.Uy = self.ofe[4]*unit
        self.Vx = self.ofe[5]*unit
        self.C = np.arctan2(self.V0, self.U0)

        if gMethod:
            Uy, Ux = np.gradient(self.U0, axis=(1,2))
            Vy, Vx = np.gradient(self.V0, axis=(1,2))
        else:
            Ux = self.Ux
            Uy = self.Uy
            Vx = self.Vx
            Vy = self.Vy

        self.div = Ux + Vy
        self.curl = Vx - Uy

    def imshow(self, t=1, div=True, curl=True, **kwargs):
        """
        Display an data with velocity vector field.

        Parameters
        ----------
        t: int
            Default is 1.
        div: bool
            If True, display divergence map.
        curl: bool
            If True, display curl map.
        """

        try:
            plt.rcParams['keymap.back'].remove('left')
            plt.rcParams['keymap.forward'].remove('right')
        except:
            pass

        self.t = t
        self._onDiv = div
        self._onCurl = curl
        kwargs['extent'] = kwargs.pop('extent', self.extent)
        kwargs['origin'] = kwargs.pop('origin', 'lower')
        width = kwargs.pop('width', 0.004)
        scale = kwargs.pop('scale', 200)

        if div or curl:
            nw = div + curl + 1
        else:
            nw = 1
        self.nw = nw
        self.fig = plt.figure(self.otype, figsize=(6*nw,6), clear=True)
        gs = GridSpec(11, nw, wspace=0, hspace=0)

        self.axVec = self.fig.add_subplot(gs[:10, 0])
        self.axSlider = self.fig.add_subplot(gs[10, :])
        self.im = self.axVec.imshow(self.data[self.t], **kwargs)
        self.vec = self.axVec.quiver(self._xarr, self._yarr,
                                     self.U0[self.t],
                                     self.V0[self.t],
                                     self.C[self.t],
                                     cmap=plt.cm.hsv,
                                     width=width,
                                     scale=scale)

        self.axVec.set_title(r'$\mathbf{v}$')
        self.axVec.set_xlabel('X')
        self.axVec.set_ylabel('Y')
        self.axVec.set_title(r'$\mathbf{v}$ field ' f'({self.otype})')
        if div:
            self.axDiv = self.fig.add_subplot(gs[:100, 1], sharex=self.axVec,
                                              sharey=self.axVec)
            self.imDiv = self.axDiv.imshow(self.div[self.t],
                                           plt.cm.Seismic,
                                           **kwargs)
            self.axDiv.tick_params(labelbottom=False, labelleft=False)
            self.axDiv.set_title(r'$\mathbf{\nabla} \cdot$'
                                 r'$\mathbf{v}$')

        if curl:
            self.axCurl = self.fig.add_subplot(gs[:10, -1], sharex=self.axVec,
                                               sharey=self.axVec)
            self.imCurl = self.axCurl.imshow(self.curl[self.t],
                                             plt.cm.PiYG,
                                             **kwargs)
            self.axCurl.tick_params(labelbottom=False, labelleft=False)
            self.axCurl.set_title(r'$\mathbf{\nabla} \times \mathbf{V}$')

        self.sT = Slider(self.axSlider, 'Time(pix)', 0, self.nt-1,
                         valinit=self.t, valstep=1, valfmt="%i")
        self.sT.on_changed(self._chTime)
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('key_press_event', self._onKey)

    def _chTime(self, val):
        self.t = int(self.sT.val)
        self.im.set_data(self.data[self.t])
        self.vec.set_UVC(self.U0[self.t], self.V0[self.t], self.C[self.t])
        if self._onDiv:
            self.imDiv.set_data(self.div[self.t])
        if self._onCurl:
            self.imCurl.set_data(self.curl[self.t])

    def _onKey(self, event):
        if event.key == 'left':
            if self.t > 0:
                self.t -= 1
            else:
                self.t = self.nt-1
            self.sT.set_val(self.t)
        elif event.key == 'right':
            if self.t < self.nt-1:
                self.t += 1
            else:
                self.t = 0
            self.sT.set_val(self.t)




#    def runNAVE():
