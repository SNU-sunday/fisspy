from __future__ import absolute_import, division

import numpy as np
from fisspy.image.base import rot
from fisspy import cm
import astropy.units as u
import sunpy.map
from sunpy.physics.differential_rotation import rot_hpc

__author__="Juhyeong Kang"
__email__="jhkang@astro.snu.ac.kr"

__all__=["fissmap", "map_header", "align", "map_rot_correct"]

def fissmap(data0,header0,pre_align=False,**kwargs):
    """
    Make sunpy.map.Map for given data and header.
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    Example
    -------
    
    """
    if data0.ndim !=2:
        raise ValueError('Data must be 2-dimensional numpy.ndarray')
    
    if not pre_align:
        data,header=align(data0,header0)
    else:
        data=data0.copy()
        header=map_header(header0)
        header['crpix1']=header['crpix1']+header['shift1']+header['margin1']
        header['crpix2']=header['crpix2']+header['shift2']+header['margin2']
        header['crota2']=0
        
    fmap=sunpy.map.Map(data,header)
    fmap.plot_settings['title']=fmap.name.replace('Angstrom','$\AA$')
    interp=kwargs.pop('interpolation','bilinear')
    fmap.plot_settings['interpolation']=interp
    clim=kwargs.pop('clim',False)
    if clim:
        fmap.plot_settings['clim']=clim
    cmap=kwargs.pop('cmap',False)
    if cmap:
        fmap.plot_settings['cmap']=cmap
    else:
        if header['wavelen']=='6562.8':
            fmap.plot_settings['cmap']=cm.ha
        elif header['wavelen']=='8542':
            fmap.plot_settings['cmap']=cm.ca
    title=kwargs.pop('title',False)
    if title:
        fmap.plot_settings['title']=title

    return fmap
    
def map_header(header0):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    Example
    -------
    
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

def align(data0,header0,missing=0):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    Example
    -------
    
    """
    mheader=map_header(header0)
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
    
def map_rot_correct(mmap,refx,refy,reftime):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    Example
    -------
    
    """
    t=mmap.date
    if type(refx) == float:
        refx*=u.arcsec
    if type(refy) == float:
        refy*=u.arcsec
    x,y=rot_hpc(refx,refy,reftime,t)
    sx=x-refx
    sy=y-refy
    smap=mmap.shift(-sx,-sy)
    return smap