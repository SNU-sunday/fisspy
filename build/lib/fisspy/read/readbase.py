"""
"""

from __future__ import absolute_import, division
import numpy as np
from astropy.io import fits
from os.path import join, dirname


__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ["readFrame", "_readPCA", "getHeader", "getRaster"]

def readFrame(file, pfile=False, x1=0, x2=None, y1=0, y2=None, ncoeff=False):
    """
    Read the FISS fts file.

    Parameters
    ----------
    file : `str`
        File name of the FISS fts data.
    pfile : `str`
        File name of the _p.fts file of the compressed data
    x1 : `int`, optional
        A left limit index of the frame along the scan direction
    x2 : `int`, optional
        A right limit index of the frame along the scan direction
        If None, read all data from x1 to the end of the scan direction.
    y1 : `int`, optional
        A left limit index of the frame along the scan direction
    y2 : `int`, optional
        A right limit index of the frame along the scan direction
        If None, read all data from x1 to the end of the scan direction.
    noceff : `int`, optional
        he number of coefficients to be used for
        the construction of frame in a pca file.
    """
    if not x2 is None:
        if x2 > 0 and x2 <= x1:
            raise ValueError('x2 must be larger than x1')
    if not y2 is None:
        if y2 > 0 and y2 <= y1:
            raise ValueError('y2 must be larger than y1')


    if pfile:
        spec = _readPCA(file, pfile, x1=x1, x2=x2, y1=y1, y2=y2, ncoeff=ncoeff)
    else:
        if x2 is None:
            spec = fits.getdata(file)[x1:]
        else:
            spec = fits.getdata(file)[x1:x2]
        if y2 is None:
            spec = spec[:, y1:]
        else:
            spec = spec[:, y1:y2]

    spec = spec.transpose((1,0,2)).astype(float)

    return spec


def _readPCA(file, pfile, x1=0, x2=None, y1=0, y2=None, ncoeff=False):
    """
    Read the PCA compressed FISS fts file.

    Parameters
    ----------
    file : `str`
        File name of the FISS fts data.
    pfile : `str`
        File name of the _p.fts file of the compressed data
    x1 : `int`, optional
        A left limit index of the frame along the scan direction
    x2 : `int`, optional
        A right limit index of the frame along the scan direction
        If None, read all data from x1 to the end of the scan direction.
    y1 : `int`, optional
        A left limit index of the frame along the scan direction
    y2 : `int`, optional
        A right limit index of the frame along the scan direction
        If None, read all data from x1 to the end of the scan direction.
    noceff : `int`, optional
        he number of coefficients to be used for
        the construction of frame in a pca file.
    """

    pdata = fits.getdata(join(dirname(file), pfile))
    if x2 is None:
        data = fits.getdata(file)[x1:]
    else:
        data = fits.getdata(file)[x1:x2]
    if y2 is None:
        data = data[:, y1:]
    else:
        data = data[:, y1:y2]
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
