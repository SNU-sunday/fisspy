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
import os

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"
__all__ = ['manual']

use("Qt4Agg")

def manual(fiss_file,sdo_file,dirname=False,filename=False,
           wvref=-4,reflect=True,alpha=0.5,sil=True,
           missing=0):

    if type(fiss_file) == list and len(fiss_file) != 1:
        fiss_file0=fiss_file[0]
    else:
        fiss_file0=fiss_file
        
    fiss0=read.raster(fiss_file0,wvref,0.05)
    if reflect:
        fiss0=fiss0.T
    fissh=read.getheader(fiss_file0)
    xpos=fissh['tel_xpos']
    ypos=fissh['tel_ypos']
    time=fissh['date']
    
    if not filename:
        filename=time[:10]
    if not dirname:
        dirname=os.getcwd()+os.sep    
    
    filename+='_match_wcs'
    
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
        filename2=dirname+filename+'.npz'
        
        px=x0+xsld.val+xsubsld.val-xc
        py=y0+ysld.val+ysubsld.val-yc
        wcsx=px*scdelt+(nx0//2)*fcdelt
        wcsy=py*scdelt+(ny0//2)*fcdelt
        angle=major_angle.val+minor_angle.val
        
        print('save the parameter as %s'%filename2)
        np.savez(filename2,match_angle=angle,wcsx=wcsx,wcsy=wcsy)
        
    def alignb():
        print('Align the fiss data, it takes some time.')
        res=fiss_align_inform(fiss_file,wvref=wvref,dirname=dirname,
                              filename=filename,pre_match_wcs=True,
                              sil=sil,missing=missing)
        
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
