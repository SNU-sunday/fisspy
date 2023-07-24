import numpy as np
from astropy.io import fits
from glob import glob
from fisspy.image.base import alignoffset, rot

# make class (usful for debugging but should be optimized for memory)
class calFlat:
    def __init__(self):
        None
    
def make_master_flat(fflat,save_fits=True, save_fname=None, tilt=None, test=True):
    oFlat = fits.getdata(fflat)
    nf, ny, nw = oFlat.shape

    if test:
        logFlat = np.log10(oFlat.mean(0))
    else:
        logFlat = np.log10(oFlat).mean(0)

    make_slit_pattern
    return Flat

def make_slit_pattern(mflat, tilt=None, test=True, cubic=False):
    ny, nx = mflat.shape

    if tilt is None:
        tilt = get_tilt(mflat, method=1)
    
    ri = rot(mflat, np.deg2rad(tilt), cubic=cubic)

    rslit = ri[:,40:-40].mean(1)
    Slit = rot(rslit, -np.deg2rad(tilt), cubic=cubic)
    return Slit

def get_tilt(img, method=1):
    ny, nw = img.shape
    wp = 40

    if method == 1:
        dy_img = np.gradient(img, axis=0)
        i1 = dy_img[:, 40:wp+20]
        i2 = dy_img[:, -(wp+20):-40]
        shift = alignoffset(i2, i1)
        Tilt = np.rad2deg(np.arctan2(shift[0], nw - wp*2))[0]
    elif method == 2:
        dy_img = np.gradient(img, axis=0)
        i1 = dy_img[:, 40:wp+20].mean(1)
        i2 = dy_img[:, -(wp+20):-40].mean(1)
        one = np.ones(4)
        i1 = one*i1[:,None]
        i2 = one*i2[:,None]
        shift = alignoffset(i2, i1)
        Tilt = np.rad2deg(np.arctan2(shift[0], nw - wp*2))[0]

    print(f"Tilt: {Tilt}")
    return Tilt