"""
"""

from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
from os.path import join, dirname


__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ["readFrame", "_readPCA", "getHeader", "getRaster"]

def readFrame(file, pfile=False, x1=0, x2=False, ncoeff=False, xmax=False):
    """
    Read the FISS fts file.
    """
    if x2 and x2 <= x1:
        raise ValueError('x2 must be larger than x1')


    if pfile:
        spec = _readPCA(file, pfile, x1=x1, x2=x2, ncoeff=ncoeff, xmax=xmax)
    else:
        if xmax:
            spec = fits.getdata(file)
        elif not xmax:
            spec = fits.getdata(file)[x1:x1+1]
        else:
            spec = fits.getdata(file)[x1:x2]
    spec = spec.transpose((1,0,2)).astype(float)

    return spec


def _readPCA(file, pfile, x1=0, x2=False, ncoeff=False, xmax=False):
    """
    Read the PCA compressed FISS fts file.
    """

    pdata = fits.getdata(join(dirname(file), pfile))
    if xmax:
        data = fits.getdata(file)
    elif not x2:
        data = fits.getdata(file)[x1:x1+1]
    else:
        data = fits.getdata(file)[x1:x2]
    ncoeff1 = data.shape[2] - 1
    if not ncoeff:
        ncoeff = ncoeff1
    elif ncoeff > ncoeff1:
        ncoeff = ncoeff1

    spec = np.dot(data[:,:,0:ncoeff], pdata[0:ncoeff,:])
    spec *= 10.**data[:,:,ncoeff][:,:,None]
    return spec

def getHeader(file):
    """
    Get the FISS fts file header.

    Returns
    -------
    header : `astropy.io.fits.Header`
        The fts file header.

    Notes
    -----
        This function automatically check the existance of the pca file by
        reading the fts header.
    """
    header0 = fits.getheader(file)

    pfile = header0.pop('pfile',False)
    if not pfile:
        return header0
    else:
        header = fits.Header()
        header['pfile']=pfile
        for i in header0['comment']:
            sori = i.split('=')
            if len(sori) == 1:
                skv = sori[0].split(None,1)
                if len(skv) == 1:
                    pass
                else:
                    header[skv[0]] = skv[1]
            else:
                key = sori[0]
                svc = sori[1].split('/')
                try:
                    item = float(svc[0])
                except:
                    item = svc[0].split("'")
                    if len(item) != 1:
                        item = item[1].split(None,0)[0]
                    else:
                        item = item[0].split(None,0)[0]
                try:
                    if item-int(svc[0]) == 0:
                        item = int(item)
                except:
                    pass
                if len(svc) == 1:
                    header[key] = item
                else:
                    header[key] = (item,svc[1])

    header['simple'] = True
    alignl=header0.pop('alignl',-1)

    if alignl == 0:
        keys=['reflect','reffr','reffi','cdelt2','cdelt3','crota2',
              'crpix3','shift3','crpix2','shift2','margin2','margin3']
        header['alignl'] = (alignl,'Alignment level')
        for i in keys:
            header[i] = (header0[i],header0.comments[i])
        header['history'] = str(header0['history'])
    if alignl == 1:
        keys=['reflect','reffr','reffi','cdelt2','cdelt3','crota1',
              'crota2','crpix3','crval3','shift3','crpix2','crval2',
              'shift2','margin2','margin3']
        header['alignl'] = (alignl,'Alignment level')
        for i in keys:
            header[i] = (header0[i],header0.comments[i])
        header['history'] = str(header0['history'])

    return header

def getRaster(data, wave, wvPoint, wvDelt, hw=0.05):
    """
    getRaster(wv, hw)

    Make a raster image for a given wavelength with in width 2*hw

    Parameters
    ----------
    wv : float
        Referenced wavelengths.
    hw   : float
        A half-width of wavelength integration in unit of Angstrom.
        Default is 0.05

    """
    if hw < abs(wvDelt)/2.:
        hw = abs(wvDelt)/2.

    s = np.abs(wave - wvPoint) <= hw
    return data[:,:,s].mean(2)
