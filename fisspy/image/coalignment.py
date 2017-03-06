"""
Alignment FISS data.
"""
from __future__ import absolute_import, print_function, division

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"

from scipy.fftpack import ifft2,fft2
import numpy as np
from fisspy.io.read import getheader,raster
from astropy.time import Time
from sunpy.physics.differential_rotation import rot_hpc
import astropy.units as u
from astropy.io import fits
from time import clock
import os
from .base import rotation,rot_trans,rescale,rot
from shutil import copy2
import matplotlib.pyplot as plt
import fisspy
try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import *
    from fisspy.vc import qtvc
except:
    from PyQt4 import QtCore
    from PyQt4.QtGui import *
from skimage.viewer.widgets.core import Slider, Button
from sunpy.net import vso

__all__ = ['alignoffset', 'fiss_align_inform','match_wcs','manual']

def alignoffset(image0,template0):
    """
    Align the two images
    
    Parameters
    ----------
    image0 : ~numpy.ndarray
        Images for coalignment with the template
        A 2 or 3 Dimensional array ex) image[t,y,x]
    template0 : ~numpy.ndarray
        The reference image for coalignment
        2D Dimensional arry ex) template[y,x]
           
    Returns
    -------
    x : float or ~numpy.ndarray
        The single value or array of the offset values.
    y : float or ~numpy.ndarray
        The single value or array of the offset values.
    
    Notes
    -----
        This code is based on the IDL code ALIGNOFFSET.PRO
        written by J. Chae 2004.
        Using for loop is faster than inputing the 3D array as,
            >>> res=np.array([alignoffset(image[i],template) for i in range(nt)])
        where nt is the number of elements for the first axis.
        
    Example
    -------
    >>> y, x = alignoffset(image,template)
    """
    st=template0.shape
    si=image0.shape
    ndim=image0.ndim
    
    if ndim>3 or ndim==1:
        raise ValueError('Image must be 2 or 3 dimensional array.')
    
    if not st[-1]==si[-1] and st[-2]==si[-2]:
        raise ValueError('Image and template are incompatible\n'
        'The shape of image = %s\n The shape of template = %s.'
        %(repr(si[-2:]),repr(st)))
    
    if not ('float' in str(image0.dtype) and 'float' in str(template0.dtype)):
        image0=image0.astype(float)
        template0=template0.astype(float)
    
    nx=st[-1]
    ny=st[-2]
    
    template=template0.copy()
    image=image0.copy()
    
    image=(image.T-image.mean(axis=(-1,-2))).T
    template-=template.mean()
    
    sigx=nx/6.
    sigy=ny/6.
    gx=np.arange(-nx/2,nx/2,1)
    gy=np.arange(-ny/2,ny/2,1)[:,np.newaxis]    
    gauss=np.exp(-0.5*((gx/sigx)**2+(gy/sigy)**2))**0.5
    
    #give the cross-correlation weight on the image center
    #to avoid the fast change the image by the granular motion or strong flow
    
    cor=ifft2(ifft2(template*gauss)*fft2(image*gauss)).real

    # calculate the cross-correlation values by using convolution theorem and 
    # DFT-IDFT relation
    
    s=np.where((cor.T==cor.max(axis=(-1,-2))).T)
    x0=s[-1]-nx*(s[-1]>nx/2)
    y0=s[-2]-ny*(s[-2]>ny/2)
    
    if ndim==2:
        cc=np.empty((3,3))
        cc[0,1]=cor[s[0]-1,s[1]]
        cc[1,0]=cor[s[0],s[1]-1]
        cc[1,1]=cor[s[0],s[1]]
        cc[1,2]=cor[s[0],s[1]+1-nx]
        cc[2,1]=cor[s[0]+1-ny,s[1]]
        x1=0.5*(cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-2.*cc[1,1])
        y1=0.5*(cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-2.*cc[1,1])
    else:
        cc=np.empty((si[0],3,3))
        cc[:,0,1]=cor[s[0],s[1]-1,s[2]]
        cc[:,1,0]=cor[s[0],s[1],s[2]-1]
        cc[:,1,1]=cor[s[0],s[1],s[2]]
        cc[:,1,2]=cor[s[0],s[1],s[2]+1-nx]
        cc[:,2,1]=cor[s[0],s[1]+1-ny,s[2]]
        x1=0.5*(cc[:,1,0]-cc[:,1,2])/(cc[:,1,2]+cc[:,1,0]-2.*cc[:,1,1])
        y1=0.5*(cc[:,0,1]-cc[:,2,1])/(cc[:,2,1]+cc[:,0,1]-2.*cc[:,1,1])

    
    x=x0+x1
    y=y0+y1
    
    return y, x


def fiss_align_inform(file,smooth=False,**kwargs):
    """
    Calculate the fiss align information, and save to npz file.
    The reference image is the first one of the file list.
    
    Parameters
    ----------
    file : list
        The list of fts file.
    dirname : (optional) str
        The directory name for saving the npz data.
        The the last string elements must be the directory seperation
        ex) dirname='D:\\the\\greatest\\scientist\\kang\\'
        If False, the dirname is the present working directory.
    filename : (optional) str
        The file name for saving the npz data.
        There are no need to add the extension.
        If False, the filename is the date of FISS data.
    half_way : (optional) bool
        If True, use the central half image to align.
            * Default is True
    pre_match_wcs : (optional) bool
        If False, it only save the align information of FISS. (level 0)
        If True, it read the wcs file and remove it, then finally
        save the align information and wcs information to the npz file. (level1)
    save : (optional) bool
        If True, save the align information.
            * Default is True.
    sil : (optional) bool
        If False, it print the ongoing time index.
            * Default is True.
    sol_rot : (optional) bool
        If True, correct the solar rotation when update the file header.
            * Default is False.
    update_header : (optional) bool
        If true, update the FISS header to add the align information.
        The fts files with updated header is copyed on the match directory and
        marked by adding m to the file name.
        e.g. (/data/*_c.fts -> /data/match/*_cm.fts)
            * Defualt is True.
    wvref : (optional) float
        The referenced wavelength for making raster image.
    kwargs
        Any additional keyword arguments to pass the `fisspy.io.read.raster`
        and to insert the `fisspy.image.coalignment.update_fiss_header`
    
    Returns
    -------
    npz file.
    xc : float
        Central position of image.
    yc : float
        Central position of image.
    angle : ~numpy.ndarray
        The array of align angle.
    dt : ~numpy.ndarray
        The array of time difference for reference image.
    dx : ~numpy.ndarray
        The relative displacement along x-axis 
        of the rotated images to the reference image.
    dy : ~numpy.ndarray
        The relative displacement along y-axis 
        of the rotated images to the reference image.
    
    Notes
    -----
    This code is based on the IDL code FISS_ALIGN_DATA.PRO written by J. Chae 2015
    
    The dirname must be have the directory seperation.
    
    Example
    -------
    >>> from glob import glob
    >>> from fisspy.image import coalignment
    >>> file=glob('*_A1_c.fts')
    >>> dirname='D:\\im\\so\\hot\\'
    >>> coalignment.fiss_align_inform(file,dirname=dirname,sil=False)
    
    """
    t0=clock()
    sil=kwargs.get('sil',False)
    half_way=kwargs.pop('half_way',True)
    
    if not sil:
        print('====== Fiss Alignment ======')
        
    n=len(file)
    ref_frame=kwargs.pop('ref_frame',n//2)
    wvref=kwargs.pop('wvref',-4)
    save=kwargs.pop('save',True)
    dirname=kwargs.pop('dirname',os.getcwd()+os.sep)
    
    
    hlist=[getheader(i) for i in file]
    tlist=[i['date'] for i in hlist]
    t=Time(tlist,format='isot',scale='ut1')
    dtmin=(t.jd-t.jd[ref_frame])*24*60
    
           
           
    filename=kwargs.pop('filename',t[0].value[:10]+'_'+hlist[0]['wavelen'][:4])
    pre_match_wcs=kwargs.pop('pre_match_wcs',False)
    reflect=kwargs.pop('reflect',True)
    update_header=kwargs.pop('update_header',True)
    
    nx=hlist[0]['naxis3']
    ny=hlist[0]['naxis2']
    
    angle=np.deg2rad(dtmin*0.25)
    
    xc=nx//2
    yc=ny//2
    
    if half_way:
        nx1=nx//2
        ny1=ny//2
        nx1=(nx1//2)*2
        ny1=(ny1//2)*2
        x1=xc-nx1//2
        y1=yc-ny1//2
    else:
        x1=1
        y1=1
        nx1=nx-1
        ny1=ny-1
    
    xa=(x1+np.arange(nx1))
    ya=(y1+np.arange(ny1))[:,None]
    im1=raster(file[ref_frame],wvref,0.05,x1,x1+nx1,y1,y1+ny1,smooth=smooth,
               **kwargs)
    dx=np.zeros(n)
    dy=np.zeros(n)
    

    print('The reference frame is %d'%ref_frame)
    if ref_frame==0:
        for i in range(n-1):
            #align with next image
            im2=raster(file[i+1],wvref,0.05,x1,x1+nx1,y1,y1+ny1,smooth=smooth,
                       **kwargs)
            img1=rotation(im1,angle[i],xa,ya,xc,yc,missing=-1)
            img2=rotation(im2,angle[i+1],xa,ya,xc,yc,missing=-1)
            sh=alignoffset(img2,img1)
            
            #align with reference
            img1=rotation(im1,angle[i],xa,ya,xc,yc,dx[i],dy[i],missing=-1)
            img2=rotation(im2,angle[i+1],xa,ya,xc,yc,dx[i]+sh[1],dy[i]+sh[0],missing=-1)
            sh+=alignoffset(img2,img1)
            dx[i+1]=dx[i]+sh[1]
            dy[i+1]=dy[i]+sh[0]
            
            im1=im2
            
            if not sil:
                print(i)
    else:
        for i in range(ref_frame,n-1):
            #align with next image
            im2=raster(file[i+1],wvref,0.05,x1,x1+nx1,y1,y1+ny1,smooth=smooth,
                       **kwargs)
            img1=rotation(im1,angle[i],xa,ya,xc,yc,missing=-1)
            img2=rotation(im2,angle[i+1],xa,ya,xc,yc,missing=-1)
            sh=alignoffset(img2,img1)
            
            #align with reference
            img1=rotation(im1,angle[i],xa,ya,xc,yc,dx[i],dy[i],missing=-1)
            img2=rotation(im2,angle[i+1],xa,ya,xc,yc,dx[i]+sh[1],dy[i]+sh[0],missing=-1)
            sh+=alignoffset(img2,img1)
            dx[i+1]=dx[i]+sh[1]
            dy[i+1]=dy[i]+sh[0]
            
            im1=im2
            if not sil:
                print(i)
        im1=raster(file[ref_frame],wvref,0.05,x1,x1+nx1,y1,y1+ny1,
                   smooth=smooth,**kwargs)
        for i in range(ref_frame,0,-1):
            #align with next image
            im2=raster(file[i-1],wvref,0.05,x1,x1+nx1,y1,y1+ny1,smooth=smooth,
                       **kwargs)
            img1=rotation(im1,angle[i],xa,ya,xc,yc,missing=-1)
            img2=rotation(im2,angle[i-1],xa,ya,xc,yc,missing=-1)
            sh=alignoffset(img2,img1)
            
            #align with reference
            img1=rotation(im1,angle[i],xa,ya,xc,yc,dx[i],dy[i],missing=-1)
            img2=rotation(im2,angle[i-1],xa,ya,xc,yc,dx[i]+sh[1],dy[i]+sh[0],missing=-1)
            sh2=alignoffset(img2,img1)
            dx[i-1]=dx[i]+sh[1]+sh2[1]
            dy[i-1]=dy[i]+sh[0]+sh2[0]
            
            im1=im2
            if not sil:
                print(i)
    if not sil:
        print('end loop')
    result=dict(xc=xc,yc=yc,angle=angle,dt=dtmin,dx=dx,dy=dy)
    if save:
        filename2=dirname+filename
        if not pre_match_wcs:
            if not sil:
                print('You select the no pre_match_wcs')
            fileout=filename2+'_align_lev0.npz'
            np.savez(fileout,
                     xc=xc,
                     yc=yc,
                     angle=angle,
                     dt=dtmin,
                     dx=dx,
                     dy=dy,
                     time=t,
                     reflect=False,
                     reffr=ref_frame,
                     reffi=file[ref_frame])
        else:
            fileout=filename2+'_align_lev1.npz'
            if not sil:
                print('You select the pre_match_wcs')
                print(filename2)
            tmp=np.load(filename2+'_match_wcs.npz')
            if reflect:
                angle=np.deg2rad(tmp['match_angle'])-angle
            else:
                angle+=np.deg2rad(tmp['match_angle'])
            np.savez(fileout,
                     xc=xc,
                     yc=yc,
                     angle=angle,
                     dt=dtmin,
                     dx=dx,
                     dy=dy,
                     sdo_angle=np.deg2rad(tmp['match_angle']),
                     wcsx=tmp['wcsx'],
                     wcsy=tmp['wcsy'],
                     reflect=reflect,
                     reffr=ref_frame,
                     reffi=file[ref_frame])
        if not sil:
            print('The saving file name is %s'%fileout)
    if not sil:
        print('The running time is %.2f seconds'%(clock()-t0))
        
    if update_header:
        update_fiss_header(file,fileout,**kwargs)
    return result

def update_fiss_header(file,alignfile,**kwargs):
    """
    Create the new FISS data with update header.
    
    Parameters
    ----------
    file : list
        FISS fts file list
    alignfile : str
        Aligned information binary (.npz) file.
    sil : (optional) bool
        If False, it print the ongoing time index.
            * Default is True.
    sol_rot : (optional) bool
        If True, correct the solar rotation when update the file header.
            * Default is False.
        
    Returns
    -------
    mfts : file
        List of files which updated the header.
        
    Notes
    -----
        Name of saved fits data file is added 'm' to 
        the first character of original data name.
    """
    sil=kwargs.pop('sil',False)
    sol_rot=kwargs.pop('sol_rot',False)
    
    if not sil:
        print('Add the align information to the haeder.')
    level=alignfile[-8:-4]
    inform=np.load(alignfile)
    fissht=[getheader(i) for i in file]
    fissh=[fits.getheader(i) for i in file]
    tlist=[i['date'] for i in fissht]
    
    time=Time(tlist,format='isot',scale='ut1')
    angle=inform['angle']
    ny=fissht[0]['naxis2']
    nx=fissht[0]['naxis3']
    x=np.array((0,nx-1,nx-1,0))
    y=np.array((0,0,ny-1,ny-1))
    xc=inform['xc'].item()
    yc=inform['yc'].item()
    dx=inform['dx']
    dy=inform['dy']
    
    xt1,yt1=rot_trans(x,y,xc,yc,angle.max())
    xt2,yt2=rot_trans(x,y,xc,yc,angle.min())
    
    tmpx=np.concatenate((xt1,xt2))
    tmpy=np.concatenate((yt1,yt2))
    
    xmargin=int(np.abs(np.round(tmpx.min()+dx.min())))+1
    ymargin=int(np.abs(np.around(tmpy.min()+dy.min())))+1
    
    if level=='lev0':
        for i,h in enumerate(fissh):
            h['alignl']=(0,'Alignment level')
            h['reflect']=(False,'Mirror reverse')
            h['reffr']=(inform['reffr'].item(),'Reference frame in alignment')
            h['reffi']=(inform['reffi'].item(),'Reference file name in alignment')
            h['cdelt2']=(0.16,'arcsec per pixel')
            h['cdelt3']=(0.16,'arcsec per pixel')
            h['crota2']=(angle[i],
                        'Roation angle about reference pixel')
            h['crpix3']=(inform['xc'].item(),'Reference pixel in data axis 3')
            h['shift3']=(inform['dx'][i],
                        'Shifting pixel value along data axis 2')
            h['crpix2']=(inform['yc'].item(),'Reference pixel in data axis 2')
            h['shift2']=(inform['dy'][i],
                        'Shifting pixel value along data axis 3')
            h['margin2']=(ymargin,'Rotation margin in axis 2')
            h['margin3']=(xmargin,'Rotation margin in axis 3')
            
            h['history']='FISS aligned (lev0)'
    elif level=='lev1':
        wcsx=inform['wcsx']
        wcsy=inform['wcsy']
        xref=wcsx*u.arcsec
        yref=wcsy*u.arcsec
        reffr=inform['reffr']
        for i,h in enumerate(fissh):
            if sol_rot:
                wcsx, wcsy=rot_hpc(xref,yref,time[reffr],time[i])
                h['crval3']=(wcsx.value,
                            'Location of ref pixel x (arcsec)')
                h['crval2']=(wcsy.value,
                            'Location of ref pixel y (arcsec)')
            else:
                h['crval3']=(wcsx.item(),
                            'Location of ref pixel for ref frame x (arcsec)')
                h['crval2']=(wcsy.item(),
                            'Location of ref pixel for ref frame y (arcsec)')

            h['alignl']=(1,'Alignment level')
            h['reflect']=(inform['reflect'].item(),'Mirror reverse')
            h['reffr']=(inform['reffr'].item(),'Reference frame in alignment')
            h['reffi']=(inform['reffi'].item(),'Reference file name in alignment')
            h['cdelt2']=(0.16,'arcsec per pixel')
            h['cdelt3']=(0.16,'arcsec per pixel')
            h['crota1']=(inform['sdo_angle'].item(),
                        'Rotation angle of reference frame (radian)')
            h['crota2']=(inform['angle'][i],
                        'Rotation angle about reference pixel (radian)')
            h['crpix3']=(inform['xc'].item(),'Reference pixel in data axis 3')
            h['shift3']=(inform['dx'][i],
                        'Shifting pixel value along data axis 3')
            h['crpix2']=(inform['yc'].item(),'Reference pixel in data axis 2')

            h['shift2']=(inform['dy'][i],
                        'Shifting pixel value along data axis 2')
            h['margin2']=(ymargin,'Rotation margin in axis 2')
            h['margin3']=(xmargin,'Rotation margin in axis 3')
            h['srot']=(True,'Solar Rotation correction')
            h['history']='FISS aligned and matched wcs (lev1)'
    else:
        raise ValueError('The level of alignfile is neither lev0 or lev1.')
    
    data=[fits.getdata(i) for i in file]
    
    odirname=os.path.dirname(file[0])
    if not odirname:
        odirname=os.getcwd()
    dirname=odirname+os.sep+'match'
    
    try:
        os.mkdir(dirname)
    except:
        pass
    
    for i,oname in enumerate(file):
        name='m'+os.path.basename(oname)
        fits.writeto(dirname+os.sep+name,data[i],fissh[i])
    try:
        pfilelist=[i['pfile'] for i in fissh]
        pfileset=set(pfilelist)
        for i in pfileset:
            copy2(odirname+os.sep+i,dirname+os.sep+i)
    except:
        pass
    
    if not sil:
        print("The align information is updated to the header, "
              "and new fts file is locate %s the file name is '*_cm.fts'"%dirname)
    

def match_wcs(fiss_file,smooth=False,**kwargs):
    """
    Match the wcs information of FISS files with the SDO/HMI file.
    
    Parameters
    ----------
    fiss_file : str or list
        A single of fiss file or the list of fts file.
    sdo_file : (optional) str
        A SDO/HMI data to use for matching the wcs.
        If False, then download the HMI data on the VSO site.
    dirname : (optional) str
        The directory name for saving the npz data.
        The the last string elements must be the directory seperation.
        ex) dirname='D:\\the\\greatest\\scientist\\kang\\'
        If False, the dirname is the present working directory.
    filename : (optional) str
        The file name for saving the npz data.
        There are no need to add the extension.
        If False, the filename is the date of FISS data.
    sil : (optional) bool
        If False, it print the ongoing time index.
        Default is True.
    sdo_path : (optional) str
        The directory name for saving the HMI data.
        The the last string elements must be the directory seperation.
    method : (optioinal) bool
        If True, then manually match the wcs.
        If False, you have a no choice to this yet. kkk.
    wvref : (optional) float
        The referenced wavelength for making raster image.
    reflect : (optional) bool
        Correct the reflection of FISS data.
        Default is True.
    alpha : (optional) float
        The transparency of the image to 
    missing : (optional) float
        The extrapolated value of interpolation.
        Default is -1.
        
    Returns
    -------
    match_angle : float
        The angle to rotate the FISS data to match the wcs information.
    wcsx : float
        The x-axis value of image center in WCS arcesec unit.
    wcsy : float
        The y-axis value of image center in WCS arcesec unit.
    
    Notes
    -----
    The dirname and sdo_path must be have the directory seperation.
    
    Example
    -------
    >>> from glob import glob
    >>> from fisspy.image import coalignment
    >>> file=glob('*_A1_c.fts')
    >>> dirname='D:\\im\\so\\hot\\'
    >>> sdo_path='D:\\im\\sdo\\path\\'
    >>> coalignment.match_wcs(file,dirname=dirname,sil=False,
                              sdo_path=sdo_path)
    
    """
    ref_frame=kwargs.get('ref_frame',len(fiss_file)//2)
    kwargs['ref_frame']=ref_frame
    fiss_file0=fiss_file[ref_frame]
    sdo_file=kwargs.pop('sdo_file',False)
    sil=kwargs.get('sil',False)
    sdo_path=kwargs.pop('sdo_path',os.getcwd()+os.sep)
    
    if not sdo_file:
        h=getheader(fiss_file0)
        tlist=h['date']
        t=Time(tlist,format='isot',scale='ut1')
        tjd=t.jd
        t1=tjd-22/24/3600
        t2=tjd+22/24/3600
        t1=Time(t1,format='jd')
        t2=Time(t2,format='jd')
        t1.format='isot'
        t2.format='isot'
        hmi=(vso.attrs.Instrument('HMI') &
             vso.attrs.Time(t1.value,t2.value) &
             vso.attrs.Physobs('intensity'))
        vc=vso.VSOClient()
        res=vc.query(hmi)
        if not sil:
            print('Download the SDO/HMI file')
        sdo_file=(vc.get(res,path=sdo_path+'{file}',methods=('URL-FILE','URL')).wait())
        sdo_file=sdo_file[0]
        if not sil:
            print('SDO/HMI file name is %s'%sdo_file)
    manual(fiss_file,sdo_file,**kwargs)
    return

def manual(fiss_file,sdo_file,smooth=False,**kwargs):
    """
    Align the FISS data with the SDO maually.
    """
    ref_frame=kwargs.get('ref_frame',len(fiss_file)//2)
    fiss_file0=fiss_file[ref_frame]
    wvref=kwargs.get('wvref',-4)
    reflect=kwargs.get('reflect',True)
    
    fiss0=raster(fiss_file0,wvref,0.05,smooth=smooth,**kwargs)
    if reflect:
        fiss0=fiss0.T
        
    fissh=getheader(fiss_file0)
    xpos=fissh['tel_xpos']
    ypos=fissh['tel_ypos']
    time=fissh['date']
    wavelen=fissh['wavelen']
    
    filename=kwargs.get('filename',time[:10]+'_'+wavelen[:4])
    dirname=kwargs.get('dirname',os.getcwd()+os.sep)
    alpha=kwargs.pop('alpha',0.5)
    interp=kwargs.pop('interpolation','bilinear')
    
    sdo=fits.getdata(sdo_file)
    sdoh=fits.getheader(sdo_file)
    sdoc=np.median(sdo[2048-150:2048+150,2048-150:2048+150])
    xc=sdoh['crpix1']-1
    yc=sdoh['crpix2']-1
    sdor=rot(np.nan_to_num(sdo/sdoc),np.deg2rad(sdoh['crota2']),
             xc=xc,yc=yc)
    
    scdelt=sdoh['cdelt1']
    fcdelt=0.16
    ratio=fcdelt/scdelt
    reshape=(np.array(fiss0.shape)*ratio).astype(int)
    
    fiss=rescale(fiss0,reshape)
    
    ny0,nx0=fiss0.shape
    ny,nx=fiss.shape
  
    x0=int(xpos/scdelt+xc)
    y0=int(ypos/scdelt+yc)
    
    sdo1=sdor[y0-150+reshape[0]//2:y0+150+reshape[0]//2,
              x0-150+reshape[1]//2:x0+150+reshape[1]//2]
    extent=[(x0-xc-150+reshape[1]//2)*scdelt,(x0-xc+150+reshape[1]//2)*scdelt,
            (y0-yc-150+reshape[0]//2)*scdelt,(y0-yc+150+reshape[0]//2)*scdelt]
    extent1=[(x0-xc)*scdelt,(x0-xc+nx)*scdelt,
             (y0-yc)*scdelt,(y0-yc+ny)*scdelt]
    fig, ax=plt.subplots(1,1,figsize=(10,8))
    
    
    im1 = ax.imshow(fiss,cmap=fisspy.cm.ha,origin='lower',extent=extent1,
                    interpolation=interp)
    ax.set_xlabel('X (arcsec)')
    ax.set_ylabel('Y (arcsec)')
    ax.set_title(time)
    im2 = ax.imshow(sdo1,origin='lower',cmap=plt.cm.Greys,
                    alpha=alpha,extent=extent,interpolation=interp)
    im2.set_clim(0.6,1)
    
    def update_angle():
        angle = np.deg2rad(major_angle.val+minor_angle.val)
        tmp=rot(fiss0,angle)
        img=rescale(tmp,reshape)
        im1.set_data(img)
        fig.canvas.draw_idle()
    
    def update_xy():
        x=x0+xsld.val
        y=y0+ysld.val
        x1=x0+xsld.val+xsubsld.val
        y1=y0+ysld.val+ysubsld.val
        extent=[(x-xc-150+reshape[1]//2)*scdelt,
                (x-xc+150+reshape[1]//2)*scdelt,
                (y-yc-150+reshape[0]//2)*scdelt,
                (y-yc+150+reshape[0]//2)*scdelt]
        extent1=[(x1-xc)*scdelt,(x1-xc+nx)*scdelt,
             (y1-yc)*scdelt,(y1-yc+ny)*scdelt]
        sdo2=sdor[y-150+reshape[0]//2:y+150+reshape[0]//2,
                  x-150+reshape[1]//2:x+150+reshape[1]//2]
        im2.set_data(sdo2)
        im1.set_extent(extent1)
        im2.set_extent(extent)
        fig.canvas.draw_idle()
        
    def printb():
        px=x0+xsld.val+xsubsld.val-xc-0.5
        py=y0+ysld.val+ysubsld.val-yc-0.5
        wcsx=px*scdelt+(nx0//2)*fcdelt
        wcsy=py*scdelt+(ny0//2)*fcdelt
        angle=major_angle.val+minor_angle.val
        print('==============  Results  ===============')
        print('angle = %.2f'%angle)
        print('wcs x of image center = %.2f'%wcsx)
        print('wcs y of image center = %.2f'%wcsy)
        print('========================================')
        
    def saveb():
        filename2=dirname+filename+'_match_wcs.npz'
        
        px=x0+xsld.val+xsubsld.val-xc-0.5
        py=y0+ysld.val+ysubsld.val-yc-0.5
        wcsx=px*scdelt+(nx0//2)*fcdelt
        wcsy=py*scdelt+(ny0//2)*fcdelt
        angle=major_angle.val+minor_angle.val
        
        print('save the parameter as %s'%filename2)
        np.savez(filename2,match_angle=angle,wcsx=wcsx,
                 wcsy=wcsy,reflect=reflect)
        
    def alignb():
        print('Align the fiss data, it takes some time.')
        res=fiss_align_inform(fiss_file,pre_match_wcs=True,**kwargs)
        del res
        
    root = fig.canvas.manager.window
    panel = QWidget()
    vbox = QVBoxLayout(panel)
    major_angle=Slider('Angle',0,359,0,value_type='int')
    minor_angle=Slider('Sub-Angle',0,1.,0,value_type='float')
    xsld=Slider('X',-150,150,0,value_type='int')
    ysld=Slider('Y',-150,150,0,value_type='int')
    xsubsld=Slider('Sub-X',-1.,1.,0,value_type='float')
    ysubsld=Slider('Sub-Y',-1.,1.,0,value_type='float')
    major_angle.slider.valueChanged.connect(update_angle)
    minor_angle.slider.valueChanged.connect(update_angle)
    xsld.slider.valueChanged.connect(update_xy)
    xsubsld.slider.valueChanged.connect(update_xy)
    ysld.slider.valueChanged.connect(update_xy)
    ysubsld.slider.valueChanged.connect(update_xy)
    
    vbox.addWidget(major_angle)
    vbox.addWidget(minor_angle)
    vbox.addWidget(xsld)
    vbox.addWidget(xsubsld)
    vbox.addWidget(ysld)
    vbox.addWidget(ysubsld)
    
    hbox = QHBoxLayout(panel)
    printbt=Button('print',printb)
    savebt=Button('save',saveb)
    alignbt=Button('run align',alignb)
    hbox.addWidget(printbt)
    hbox.addWidget(savebt)
    
    vbox2 = QVBoxLayout(panel)
    vbox2.addWidget(alignbt)
    
    vbox.addLayout(hbox)
    vbox.addLayout(vbox2)
    panel.setLayout(vbox)
    dock = QDockWidget("Align Control Panel", root)
    root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    dock.setWidget(panel)
        
    plt.show()
