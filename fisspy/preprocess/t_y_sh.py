import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from fisspy.align import alignOffset
from astropy.time import Time
from scipy.signal import find_peaks

def cal_shift(fl, fref):
    """
    Calcuate the image shift

    Parameters
    ----------
    fl: `list`
        Series of the files.
    fref: `int`
        Reference frame.

    Returns
    -------
    tt: `~numpy.ndarray`
        Series of the time in JD in the unit of second
    sh:
        `~numpy.ndarray`
        Series of the shift along the slit direciton.
    date: `str`
        Datetime at the reference time.
    """
    fl.sort()
    nf = len(fl)
    opn = fits.open(fref)[0]
    fh = opn.header
    rdata = opn.data[3]
    rd2y = np.gradient(np.gradient(rdata,axis=0), axis=0)
    wp = 40
    k = rd2y[20:-20,wp:wp+20].mean(1)
    pks = find_peaks(k, k.std()*2)[0]+20
    npks = len(pks)
    
    tt = np.zeros(nf)
    mask = np.ones(nf, dtype=bool)
    sh = np.zeros(nf)
    init = True
    for i, f in enumerate(fl):
        if f.find('BiasDark') > -1:
            bd = fits.getdata(f)
            mask[i] = False
            continue
        print(i)
        opn = fits.open(f)[0]
        data = opn.data.mean(0) - bd
        tt[i] = Time(opn.header['date']).jd*24*60
        
        d2y = np.gradient(np.gradient(data - bd,axis=0), axis=0)
        tsh = 0
        for whd in pks:
            rimg = rd2y[whd-16:whd+16, 10:-10]
            img = d2y[whd-16:whd+16, 10:-10]
            tsh += alignOffset(img, rimg)[0,0]
        tsh /= npks
        sh[i] = tsh
    tt = tt[mask]
    sh = sh[mask]
    # coeff = np.polyfit(tt, sh, 3)
    return tt, sh, fh['date']

# def fitWref()
