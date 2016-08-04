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

