"""
FISS data module

Read FISS data or its PCA file.

Based on the IDL code FISS_READ_FRAME and
FISS_PCA_READ written by (J. Chae May 2013)

===========================
Including functions are
    frame
    pca_read
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


def frame(file,x1,x2=False,pca=True,ncoeff=False):
    """
    FISS READ FRAME
    
    Using the astropy package
    
    Based on the IDL code FISS_READ_FRAME written by (J. Chae 2033)
    
    Arguments
        file : string of file name to be read
        x1   : the frame number along the scanning direction
        x2   : the end of the frame number (optional)
    Keywords
        pca  : if set, data are read from the PCA file
               default default is set
        ncoeff : number of coefficients to be used for the construction of data
                 in a pca file
    ===========================================================      
    example
    >>> import fisspy
    >>> data=fisspy.read.frame(file,70,100,ncoeff=10)
    """
    global header
    
    if not file:
        raise ValueError('Empty filename: %s' % repr(file))
    
    header=fits.getheader(file)
    
    if not x2:
        x2=x1+1
    
    if pca:
        spec=pca_read(file,header,x1,x2,ncoeff=ncoeff)
    else:        
        spec=fits.getdata(file)[x1:x2]
        
    return spec.astype(float)


def pca_read(file,header,x1,x2=False,ncoeff=False):
    """
    FISS READ PCA FILE
    
    Using the astropy package
    
    Based on the IDL code FISS_PCA_READ written by (J. Chae 2013)
    
    Arguments
        file : string of file name to be read
        header : the fts file header
        x1   : the frame number along the scanning direction
        x2   : the end of the frame number (optional)
        
    Keywords
        ncoeff : number of coefficients to be used for the construction of data
                 in a pca file
                 
    """
    if not file:
        raise ValueError('Empty filename: %s' % repr(file))
    if not x2:
        x2=x1+1
        
    dir=os.path.dirname(file)
    pfile=header['pfile']
    
    if bool(dir):
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
    
    header=getheader(file)
    nw=header['NAXIS1']
    ny=header['NAXIS2']
    nx=header['NAXIS3']
    wc=header['CRPIX1']
    dldw=header['CDELT1']
    
    num=wv.shape[0]
    
    if not x2:
        x2=int(nx)
    if not y2:
        y2=int(ny)
    
    wl=(np.arange(nw)-wc)*dldw
    if hw < abs(dldw)/2.:
        hw=abs(dldw)/2.
    
    s=np.abs(wl-wv[:,np.newaxis])<=hw
    sp=frame(file,x1,x2)
    leng=s.sum(1)
    img=np.array([])
    for i in range(num):
        img=np.append(img,sp[:,y1:y2,s[i,:]].sum(2)/leng[i])
    img=img.reshape((num,x2-x1,y2-y1)).T
    return img


def getheader(file,pca=True):
    header0=fits.getheader(file)
    if pca:
        header={}
        for i in header0['comment']:
            tmp=i.split(maxsplit=3)
            if len(tmp) == 4:
                try:
                    header[tmp[0]]=float(tmp[2])
                except:
                    header[tmp[0]]=tmp[2]
            if tmp[0] == 'WAVELEN':
                header[tmp[0]]=tmp[2][1:]
    else:
        header=header0
    return header
