"""
FISS data module

Read the FISS data

===========================
Including functions are
    frame
    pca_read
    raster
    getheader
===========================

Only input the original fts file or its PCA file '*_c.fts'.

=============================================
Example
>>> import fisspy
>>> import glob
>>> file=glob.glob('*_c.fts')
>>> data=fisspy.read.frame(file[0],70,80)
=============================================
"""
from __future__ import absolute_import, division, print_function


__date__="Aug 04 2016"
__author__="J. Kang : jhkang@astro.snu.ac.kr"

from astropy.io import fits
import numpy as np
import os.path


def frame(file,x1=0,x2=False,pca=True,ncoeff=False,xmax=False):
    """
    FISS READ FRAME

    Parameters
        file : string of file name to be read
        x1   : the frame number along the scanning direction
        x2   : the end of the frame number (optional)
    Keywords
        pca  : if set, data are read from the PCA file
               default default is set
        ncoeff : number of coefficients to be used for the construction of data
                 in a pca file
    ===========================================================      
    Example)
    >>> import fisspy
    >>> data=fisspy.read.frame(file,70,100,ncoeff=10)
    ===========================================================
        
    Using the astropy package
    
    Based on the IDL code FISS_READ_FRAME written by (J. Chae 2013)
    """
    if not file:
        raise ValueError('Empty filename')
    if x2 and x2 <= x1:
        raise ValueError('x2 must be larger than x1')
    
    header=fits.getheader(file)
    
    try:
        header['pfile']
    except:
        pca=False
    
    if xmax:
        x2=header['naxis3']
    elif not x2:
        x2=x1+1
    
    if pca:
        spec=pca_read(file,header,x1,x2,ncoeff=ncoeff)
    else:        
        spec=fits.getdata(file)[x1:x2]
    if x1+1 == x2:
        spec=spec[0]
    return spec.astype(float)


def pca_read(file,header,x1,x2=False,ncoeff=False):
    """
    FISS READ PCA FILE
    
    Parameters
        file : string of file name to be read
        header : the fts file header
        x1   : the starting frame number along the scanning direction
        x2   : the end of the frame number (optional)
        
    Keywords
        ncoeff : number of coefficients to be used for the construction of data
                 in a pca file
    
    Using the astropy package
    
    Based on the IDL code FISS_PCA_READ written by (J. Chae 2013)
    """
    if not file:
        raise ValueError('Empty filename')
    if not x2:
        x2=x1+1
        
    dir=os.path.dirname(file)
    pfile=header['pfile']
    
    if dir:
        pfile=dir+'/'+pfile
        
    pdata=fits.getdata(pfile)
    data=fits.getdata(file)[x1:x2]
    ncoeff1=data.shape[2]-1
    
    if not ncoeff:
        ncoeff=ncoeff1
    elif ncoeff > ncoeff1:
        ncoeff=ncoeff1
    
    spec=np.dot(data[:,:,0:ncoeff],pdata[0:ncoeff,:])
    spec*=10.**data[:,:,ncoeff][:,:,np.newaxis]
    return spec

def raster(file,wv,hw,x1=0,x2=False,y1=0,y2=False,pca=True):
    """
    FISS Raster
    
    Make raster images for a given file at wv of wavelength within width hw
    
    Parameters
        file : string of file name to be read
        wv   : wavelengths
        hw   : A half-width for wavelength integration
               in unit of Angstrom
        x1   : the starting frame number along the scanning direction (optional)
        x2   : the end of the frame number (optional)
        y1   : the starting slit position (optional)
        y2   : the end of the slit position (optional)
        
    Keyword
        pca : if set, data are read from the PCA file
              default default is set
    ==========================================
    Example)
    >>> import fisspy
    >>> raster=fisspy.read.raster(file[0],np.array([-1,0,1]),0.05)
    """
    header=getheader(file,pca)
    nw=header['NAXIS1']
    ny=header['NAXIS2']
    nx=header['NAXIS3']
    wc=header['CRPIX1']
    dldw=header['CDELT1']
    
    if not file:
        raise ValueError('Empty filename')
    if x2 and x2 <= x1+1:
        raise ValueError('x2 must be larger than x1+1')
    
    try:
        num=wv.shape[0]    
    except:
        num=1
        wv=np.array([wv])
    
    if not x2:
        x2=int(nx)
    if not y2:
        y2=int(ny)
    
    wl=(np.arange(nw)-wc)*dldw
    if hw < abs(dldw)/2.:
        hw=abs(dldw)/2.
    
    s=np.abs(wl-wv[:,np.newaxis])<=hw
    sp=frame(file,x1,x2,pca=pca)
    leng=s.sum(1)
    if num == 1:
        img=sp[:,y1:y2,s[0,:]].sum(2)/leng[0]
        return img.reshape((x2-x1,y2-y1)).T
    else:
        img=np.array([sp[:,y1:y2,s[i,:]].sum(2)/leng[i]] for i in range(num))
        return img.reshape((num,x2-x1,y2-y1)).T


def getheader(file,pca=True):
    header0=fits.getheader(file)
    header=fits.Header()
    try:
        header0['pfile']
    except:
        pca=False
    if pca:
        header['pfile']=header0['pfile']
        header['bscale']=header0['bscale']
        for i in header0['comment']:
            sori = i.split('=')
            if len(sori) == 1:
                skv = sori[0].split(maxsplit=1)
                if len(skv) == 1:
                    pass
                else:
                    header[skv[0]] = skv[1]
            else:
                key = sori[0]
                svc = sori[1].split('/')
                try:
                    item=float(svc[0])
#                    if item-int(svc[0]) == 0:
#                        item=int(item)
                except:
                    item=svc[0].split("'")
                    if len(item) != 1:
                        item=item[1].split(maxsplit=0)[0]
                    else:
                        item=item[0].split(maxsplit=0)[0]
                try:
                    if item-int(svc[0]) == 0:
                        item=int(item)
                except:
                    pass
                if len(svc) == 1:
                    header[key]=item
                else:
                    header[key]=(item,svc[1])
    return header
