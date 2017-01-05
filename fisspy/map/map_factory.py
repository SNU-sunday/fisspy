from __future__ import absolute_import, division

import numpy as np
from fisspy.image.base import rot
from fisspy import cm
import sunpy.map

__author__="Juhyeong Kang"
__email__="jhkang@astro.snu.ac.kr"


def Map(data0,header0):
    """
    """
    if data0.ndim !=2:
        raise ValueError('Data must be 2-dimensional numpy.ndarray')
    
    mheader=map_header(header0)
    data,header=data_resize(data0,mheader)
    
    fmap=sunpy.map.Map(data,header)
    if mheader['wavelen']=='6562.8':
        fmap.plot_settings['cmap']=cm.ha
    elif mheader['wavelen']=='8542':
        fmap.plot_settings['cmap']=cm.ca
    return fmap
    
def map_header(header0):
    """
    """
    header=header0.copy()
    header['naxis']=2
    
    if header['reflect']:
        header['naxis1']=header['naxis2']
        header['naxis2']=header['naxis3']
        header['crpix1']=header['crpix2']
        header['crpix2']=header['crpix3']
        header['crval1']=header['crval3']
        header['cdelt1']=header['cdelt2']
        header['cdelt2']=header['cdelt3']
        header['shift1']=header['shift2']
        header['shift2']=header['shift3']
        header['crota2']=np.rad2deg(header['crota2'])
        header['margin1']=header['margin2']
        header['margin2']=header['margin3']
        header.remove('naxis3')
        header.remove('crpix3')
        header.remove('crval3')
        header.remove('cdelt3')
        header.remove('crota1')
        header.remove('shift3')
    else:
        header['naxis1']=header['naxis3']
        header['crpix1']=header['crpix3']
        header['crval1']=header['crval3']
        header['cdelt1']=header['cdelt3']
        header['shift1']=header['shift3']
        header['crota2']=np.rad2deg(header['crota2'])
        header['margin1']=header['margin3']
        header.remove('naxis3')
        header.remove('crpix3')
        header.remove('crval3')
        header.remove('cdelt3')
        header.remove('crota1')
        header.remove('shift3')
    
    header['instrume']='FISS'
    header['detector']='FISS'
    header['telescop']='NST'
    header['wavelnth']=float(header['wavelen'])
    header['waveunit']='Angstrom'
    header['date-obs']=header['date']
    return header 

def data_resize(data0,mheader0):
    """
    """
    mheader=mheader0.copy()
    reflect=mheader['reflect']
    
    if reflect:
        data=data0.T
        
    xmargin=mheader['margin1']
    ymargin=mheader['margin2']
    
    
    img=rot(data,np.deg2rad(mheader['crota2']),mheader['crpix1'],
            mheader['crpix2'],mheader['shift1'],mheader['shift2'],
            mheader['margin1'],mheader['margin2'])
    
    mheader['crpix1']=mheader['crpix1']+mheader['shift1']+xmargin
    mheader['crpix2']=mheader['crpix2']+mheader['shift2']+ymargin
    mheader['crota2']=0
    
    return img, mheader
    