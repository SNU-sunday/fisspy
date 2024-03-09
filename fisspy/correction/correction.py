import numpy as np
from fisspy.align.base import AlignOffset

def get_InstShift(data, refSpec, dw):
    """
    Get offsert value of the instrumental shift caused by the seeing and vibration of the spectrograph.

    Parameters
    ----------
    data: `~numpy.ndarray`
        N-d data.
        The last dimension should be the wavelength.
    refSpec: `~numpy.ndarray`
        1-D Reference spectrum.
    dw: `float`
        Wavelength scale in unit of angstrom.
    """
    ndim = data.ndim
    sh = data.shape
    if ndim >= 4:
        raise ValueError("The dimension of the data should be less than 4.")
    else:
        refSpec2der = np.gradient(np.gradient(refSpec))
        refSpec2der[:2] = refSpec2der[-2:] = 0
        if ndim == 3:
            refSpec2der = refSpec2der * np.ones(sh[1:])
            data2der = np.gradient(np.gradient(data.mean(0), axis=1), axis=1)
            data2der[:,:2] = data2der[:,-2:] = 0
        elif ndim == 2:
            refSpec2der = refSpec2der * np.ones(sh)
            data2der = np.gradient(np.gradient(data, axis=1), axis=1)
            data2der[:,:2] = data2der[:,-2:] = 0
        elif ndim == 1:
            refSpec2der = refSpec2der * np.ones((4, sh[0]))
            data2der = np.gradient(np.gradient(data))
            data2der[:2] = data2der[-2:] = 0
            data2der = data2der * np.ones((4, sh[0]))
        yoff, xoff = AlignOffset(data2der, refSpec2der)
        woffset = xoff*dw
    return woffset