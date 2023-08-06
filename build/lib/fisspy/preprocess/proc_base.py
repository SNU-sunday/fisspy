import numpy as np
from astropy.io import fits
from glob import glob
from fisspy.image.base import alignoffset, rot
import matplotlib.pyplot as plt

# make class (usful for debugging but should be optimized for memory)
class calFlat:
    def __init__(self):
        None
    
def make_master_flat(fflat,save_fits=True, save_fname=None, tilt=None, test=False, cubic=False):
    oFlat = fits.getdata(fflat)
    nf, ny, nw = oFlat.shape

    if test:
        logF = np.log10(oFlat.mean(0))
    else:
        logF = np.log10(oFlat).mean(0) # idl fiss_get_flat_v2 - line 43

    slit = make_slit_pattern(logF, tilt, cubic=cubic)
    logF -=  slit # idl fiss_get_flat_v2 - line 46
    Flat = logF
    return Flat

def make_slit_pattern(mflat, tilt=None, cubic=False):
    ny, nx = mflat.shape

    if tilt is None:
        tilt = get_tilt(mflat, method=1)
        print(f"Tilt: {tilt:.2f} degree")
    
    ri = rot(mflat, np.deg2rad(-tilt), cubic=cubic, missing=-1)

    rslit = ri[:,40:-40].mean(1)
    Slit = rot(rslit, np.deg2rad(tilt), cubic=cubic, missing=-1)
    return Slit

def get_tilt(img, show=False):
    """
    Get a tilt angle of the spectrum camera in the unit of degree.

    Parameters
    ----------
    img : `~numpy.ndarray`
        A two-dimensional `numpy.ndarray` of the form ``(y, x)``.
    show : `bool`, optional
        If `False` (default) just calculate the tilt angle.
        If `True` calculate and draw the original image and the rotation corrected image.

    Returns
    -------
    Tilt : `float`
        A tilt angle of the spectrum camera in the unit of degree.

    Examples
    --------
    >>> from astropy.io import fits
    >>> data = fits.getdata("FISS_20140603_164020_A_Flat.fts")[3]
    >>> tilt = get_tilt(data, show=True)
    """


    ny, nw = img.shape
    wp = 40

    # if method == 1:
    dy_img = np.gradient(img, axis=0)
    i1 = dy_img[:, 40:wp+20]
    i2 = dy_img[:, -(wp+20):-40]
    shift = alignoffset(i2, i1)
    Tilt = np.rad2deg(np.arctan2(shift[0], nw - wp*2))[0]
    # elif method == 2:
    #     dy_img = np.gradient(img, axis=0)
    #     i1 = dy_img[:, 40:wp+20].mean(1)
    #     i2 = dy_img[:, -(wp+20):-40].mean(1)
    #     one = np.ones(4)
    #     i1 = one*i1[:,None]
    #     i2 = one*i2[:,None]
    #     shift = alignoffset(i2, i1)
    #     Tilt = np.rad2deg(np.arctan2(shift[0], nw - wp*2))[0]

    if show:
        rimg = rot(img, np.deg2rad(-Tilt), missing=-1)
        # print(img.shape)
        whd = np.abs(dy_img[:,40:60].mean(1)).argmax()
        fig, ax = plt.subplots(1,2, figsize=[14, 4], sharey=True, sharex=True)
        imo = ax[0].imshow(img, plt.cm.gray, origin='lower', interpolation='bilinear')
        imr = ax[1].imshow(rimg, plt.cm.gray, origin='lower', interpolation='bilinear')

        ax[0].set_xlabel('Wavelength (pix)')
        ax[1].set_xlabel('Wavelength (pix)')
        ax[0].set_ylabel('Y (pix)')
        ax[0].set_title('tilted image')
        ax[1].set_title('corrected image')

        ax[0].set_ylim(whd-10,whd+10)

        ax[0].set_aspect(adjustable='box', aspect='auto')
        ax[1].set_aspect(adjustable='box', aspect='auto')

        fig.tight_layout()
        fig.show()

    # print(f"Tilt: {Tilt}")
    return Tilt