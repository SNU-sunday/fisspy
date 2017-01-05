from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
from fisspy.io import read
from fisspy import cm
from PyQt4 import QtGui, QtCore
from skimage.viewer.widgets.core import Slider, Button
from .base import rescale, rot
from .coalignment import fiss_align_inform
from astropy.io import fits
from astropy.time import Time
import os
from sunpy.net import vso

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ['match_wcs', 'manual']

use("Qt4Agg")


def match_wcs(fiss_file,sdo_file=False,dirname=False,
              filename=False,sil=True,sdo_path=False,
              method=True,wvref=-4,ref_frame=-1,reflect=True,alpha=0.5,
              missing=0,update_header=True):
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
        Default is 0.
        
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
    * The dirname and sdo_path must be have the directory seperation.
    
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
    
    if type(fiss_file) == list and len(fiss_file) != 1:
        if ref_frame==-1:
            ref_frame=len(fiss_file)//2
        fiss_file0=fiss_file[ref_frame]
    else:
        fiss_file0=fiss_file
        
    if not sdo_file:
        h=read.getheader(fiss_file0)
        tlist=h['date']
        t=Time(tlist,format='isot',scale='ut1')
        tjd=t.jd
        t1=tjd-20/24/3600
        t2=tjd+20/24/3600
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
        if not sdo_path:
            sdo_path=os.getcwd()+os.sep
        sdo_file=(vc.get(res,path=sdo_path+'{file}',methods=('URL-FILE','URL')).wait())
        sdo_file=sdo_file[0]
        if not sil:
            print('SDO/HMI file name is %s'%sdo_file)
    manual(fiss_file,sdo_file,dirname=dirname,
           filename=filename,wvref=wvref,
           reflect=reflect,alpha=alpha,ref_frame=ref_frame,
           sil=sil,missing=missing,update_header=update_header)
    return

def manual(fiss_file,sdo_file,dirname=False,filename=False,
           wvref=-4,reflect=True,alpha=0.5,ref_frame=-1,sil=True,
           missing=0,update_header=True):

    if type(fiss_file) == list and len(fiss_file) != 1:
        if ref_frame==-1:
            ref_frame=len(fiss_file)//2
        fiss_file0=fiss_file[ref_frame]
    else:
        fiss_file0=fiss_file
        
    fiss0=read.raster(fiss_file0,wvref,0.05)
    if reflect:
        fiss0=fiss0.T
    fissh=read.getheader(fiss_file0)
    xpos=fissh['tel_xpos']
    ypos=fissh['tel_ypos']
    time=fissh['date']
    wavelen=fissh['wavelen']
    
    if not filename:
        filename=time[:10]+'_'+wavelen[:4]
    if not dirname:
        dirname=os.getcwd()+os.sep    
    
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
    
    
    im1 = ax.imshow(fiss,cmap=cm.ha,origin='lower',extent=extent1)
    ax.set_xlabel('X (arcsec)')
    ax.set_ylabel('Y (arcsec)')
    ax.set_title(time)
    im2 = ax.imshow(sdo1,origin='lower',cmap=plt.cm.Greys,
                    alpha=alpha,extent=extent)
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
        px=x0+xsld.val+xsubsld.val-xc
        py=y0+ysld.val+ysubsld.val-yc
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
        
        px=x0+xsld.val+xsubsld.val-xc
        py=y0+ysld.val+ysubsld.val-yc
        wcsx=px*scdelt+(nx0//2)*fcdelt
        wcsy=py*scdelt+(ny0//2)*fcdelt
        angle=major_angle.val+minor_angle.val
        
        print('save the parameter as %s'%filename2)
        np.savez(filename2,match_angle=angle,wcsx=wcsx,
                 wcsy=wcsy,reflect=reflect)
        
    def alignb():
        print('Align the fiss data, it takes some time.')
        res=fiss_align_inform(fiss_file,wvref=wvref,ref_frame=ref_frame,
                              dirname=dirname,
                              filename=filename,pre_match_wcs=True,
                              sil=sil,missing=missing,reflect=reflect,
                              update_header=update_header)
        
    root = fig.canvas.manager.window
    panel = QtGui.QWidget()
    vbox = QtGui.QVBoxLayout(panel)
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
    
    hbox = QtGui.QHBoxLayout(panel)
    printbt=Button('print',printb)
    savebt=Button('save',saveb)
    alignbt=Button('run align',alignb)
    hbox.addWidget(printbt)
    hbox.addWidget(savebt)
    
    vbox2 = QtGui.QVBoxLayout(panel)
    vbox2.addWidget(alignbt)
    
    vbox.addLayout(hbox)
    vbox.addLayout(vbox2)
    panel.setLayout(vbox)
    dock = QtGui.QDockWidget("Align Control Panel", root)
    root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    dock.setWidget(panel)
        
    plt.show()
