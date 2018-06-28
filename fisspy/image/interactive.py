"""
Interactive
"""
#%% Importing
from __future__ import print_function, division, absolute_import
import numpy as np
from fisspy.io.read import frame, getheader
import matplotlib.pyplot as plt
from skimage.viewer.widgets.core import Slider, Button
from glob import glob
from os.path import join
from fisspy.image.coalignment import alignoffset
from fisspy.image.base import shift
import fisspy
from matplotlib.widgets import Cursor

try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import *
    from fisspy.vc import qtvc
except:
    from PyQt4 import QtCore
    from PyQt4.QtGui import *
    
__author__= "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"



#%% In code using function
def raster(frame,header,wv):
    nw=header['naxis1']
    wc=header['crpix1']
    dldw=header['cdelt1']
    hw=0.05
    wl=(np.arange(nw)-wc)*dldw
    s=np.abs(wl-wv)<=hw
    img=frame[:,:,s].sum(2)/s.sum()
    return img

def wvcalib(header):
    nw=header['naxis1']
    wc=header['crpix1']
    dldw=header['cdelt1']
    wv=(np.arange(nw)-wc)*dldw
    return wv

def colorhrange():
    hrmin=hmisld.val
    hrmax=hmasld.val
    imh.set_clim(hrmin,hrmax)
    fig.canvas.draw_idle()

def colorcrange():
    crmin=cmisld.val
    crmax=cmasld.val
    imc.set_clim(crmin,crmax)
    fig.canvas.draw_idle()

def time():
    global hframe, cframe, hh, ch, htitle, ctitle, imh, imc, ax11, ax12, scath, scatc
    hframe=frame(hflist[tsld.val],xmax=True,smooth=sm)
    hh=getheader(hflist[tsld.val])
    cframe=frame(cflist[tsld.val],xmax=True,smooth=sm)
    ch=getheader(cflist[tsld.val])
    hraster=raster(hframe,hh,wv=hwvr)
    craster=raster(cframe,ch,wv=cwvr)
    craster1=shift(craster,(-cshy,-cshx))
    x11=ax11.get_xlim()
    y11=ax11.get_ylim()
    x12=ax12.get_xlim()
    y12=ax12.get_ylim()
    ax11.remove()
    ax12.remove()
    ax11=fig.add_subplot(141)
    ax12=fig.add_subplot(142)
    imh=ax11.imshow(hraster,origin='lower',cmap=fisspy.cm.ha)
    imc=ax12.imshow(craster1,origin='lower',cmap=fisspy.cm.ca)
    imh.set_clim(hmisld.val,hmasld.val)
    imc.set_clim(cmisld.val,cmasld.val)
    ax11.set_xlim(x11)
    ax11.set_ylim(y11)
    ax12.set_xlim(x12)
    ax12.set_ylim(y12)
    ax11.set_xlabel('X position [pixel]')
    ax12.set_xlabel('X position [pixel]')
    ax11.set_ylabel('Y position [pixel]')
    ax12.set_ylabel('Y position [pixel]')
    htitle='GST/FISS '+hh['wavelen']+r' $\AA$ '+hh['date']
    ctitle='GST/FISS '+ch['wavelen']+r' $\AA$ '+ch['date']
    hi=hframe[y,x]
    ci=cframe[y+int(round(cshy[0])),x+int(round(cshx[0]))]
    p1[0].set_data(hwv,hi)
    p2[0].set_data(cwv,ci)
    ax21.set_ylim(hi.min()-100,hi.max()+100)
    ax22.set_ylim(ci.min()-100,ci.max()+100)
    ax11.set_title(htitle)
    ax12.set_title(ctitle)
    ax21.set_title(htitle)
    ax22.set_title(ctitle)
    
    try:
        scath.remove()
        scatc.remove()
    except:
        pass
    scath=ax11.scatter(x,y,marker='+',color='b')
    scatc=ax12.scatter(x,y,marker='+',color='b')
    fig.canvas.draw_idle()
    
    
def play():
    for i in range(len(hflist)):
        tsld.val=i
        plt.pause(0.01)

def pauseb():
    global pau
    pau ^= True
    if not pau:
        print('=====')
        print('pause')
        print('=====')
        fig.canvas.start_event_loop(-1)
    else:
        print('======')
        print('resume')
        print('======')
        fig.canvas.stop_event_loop()
        
        
def mark(event):
    if event.key == 'q':
        global x, y, scath, scatc, hwvr, cwvr, hwline, cwline, pau
        axp=event.inaxes._position.get_points()[0,0]
        ayp=event.inaxes._position.get_points()[0,1]
        if axp < 0.5:
            try:
                scath.remove()
                scatc.remove()
            except:
                pass
            x=int(round(event.xdata))
            y=int(round(event.ydata))
            scath=ax11.scatter(x,y,marker='+',color='b')
            scatc=ax12.scatter(x,y,marker='+',color='b')
            hi=hframe[y,x]
            ci=cframe[y+int(round(cshy[0])),x+int(round(cshx[0]))]
            p1[0].set_data(hwv,hi)
            p2[0].set_data(cwv,ci)
            ax21.set_ylim(hi.min()-100,hi.max()+100)
            ax22.set_ylim(ci.min()-100,ci.max()+100)
            
        elif ayp > 0.5:
            try:
                hwline.remove()
            except:
                pass
            hwvr=event.xdata
            hraster=raster(hframe,hh,hwvr)
            hwline=ax21.vlines(hwvr,0,1e4,linestyles='--',color='r')
            imh.set_data(hraster)

            hrmin=hraster.min()
            hrmax=hraster.max()
            hmasld.val=hrmax
            hmisld.val=hrmin
        else:
            try:
                cwline.remove()
            except:
                pass
            cwvr=event.xdata
            craster=raster(cframe,ch,cwvr)
            craster1=shift(craster,(-cshy,-cshx))
            cwline=ax22.vlines(cwvr,0,1e4,linestyles='--',color='r')
            imc.set_data(craster1)
            crmin=craster.min()
            crmax=craster.max()
            cmisld.val=crmin
            cmasld.val=crmax

        
    elif event.key == 'w':
        print('=============================================')
        print('Frame number = %i'%tsld.val)
        print('Time = %s'%hh['date'])
        print('Cam A Raster at %.2f Angstrom'%hwvr)
        print('Cam B Raster at %.2f Angstrom'%cwvr)
        print('X = %i pix'%x)
        print('Y = %i pix'%y)
        print('Align value of Cam B, x = %.2f, y = %.2f'%(cshx[0],cshy[0]))
        print('=============================================')
    elif event.key == 'e':
        for i in range(len(hflist)):
            tsld.val=i
            plt.pause(0.01)
    elif event.key == ' ' and not pau:
        print('=====')
        print('pause')
        print('=====')
        pau ^=True
        fig.canvas.start_event_loop(-1)
    elif event.key == ' ' and pau:
        print('======')
        print('resume')
        print('======')
        pau ^=True
        fig.canvas.stop_event_loop() 
    elif event.key == 'h':
        ax11.set_xlim(0,hh['naxis3'])
        ax11.set_ylim(0,hh['naxis2'])
        ax12.set_xlim(0,ch['naxis3'])
        ax12.set_ylim(0,ch['naxis2'])
    fig.canvas.draw_idle()
        
#%% IFDV
def IFDV(lista=False, listb=False, fdir=False, smooth=False):
    """
    Interactive FISS Data Viewer
    
    Plot the FISS data Interactively.
    
    The 1st panel is a raster image of CamA installed in GST/FISS.
    The 2nd one is a raster image of CamB installed in GST/FISS.
    The 3rd one is an intensity profile of the specific chosen position in the
    1st panel.
    The 4rd one is an intensity profile of the specific chosen position in the
    2st panel.
    
    ---------------------------------
    * Pressing 'q' key on the raster panels mark the position and plot 
    the intensity profile.
    * Pressing 'q' key on the intensity profile panels park the wavelength
    and re-draw the raster images at that wavelength.
    * Pressing 'w' key print the information about the frame number, time,
    position, wavelength of raster images, algined value of Cam B.
    * Pressing 'e' key play the movie. Also you can interact with this movie.
    
    
    Parameters
    ----------
    hlist : list (optional)
        The list of camera A data
        If not given, fdir must be given.
    clist : list (optional)
        The list of camera B data
        If not given, fdir must be given.
    fdir : string (optional)
        The directory for fiss file stored.
    smooth : bool (optional)
        Apply the Savitzky-Golay Filter
    """
    
    if fdir:
        clist=glob(join(fdir,'*B1*'))
        hlist=glob(join(fdir,'*A1*'))
    
    if bool(hlist) and bool(clist):
        hlist.sort()
        clist.sort()
    else:
        raise ValueError('The hlist and clist or fdir should be given.')
        
    fnum=0
    global hframe, cframe, cshy, cshx, imh, imc, p1, p2, ax11, ax12, hwv, cwv
    global ax21, ax22, hh, ch, fig, hmisld, hmasld, cmisld, cmasld
    global tsld, hflist, cflist, sm, hwvr, cwvr, pau, x, y
    pau=False
    hflist=hlist
    cflist=clist
    sm=smooth
    hwvr=4
    cwvr=4
    
    hframe=frame(hlist[fnum],xmax=True,smooth=smooth)
    hh=getheader(hlist[fnum])
    cframe=frame(clist[fnum],xmax=True,smooth=smooth)
    ch=getheader(clist[fnum])
    cshy,cshx=alignoffset(cframe[:,:,50],hframe[:250,:,-50])
    
    hraster=raster(hframe,hh,wv=hwvr)
    craster=raster(cframe,ch,wv=cwvr)
    craster1=shift(craster,(-cshy,-cshx))
    
    
        
    fig=plt.figure(figsize=(17,9))
    ax11=fig.add_subplot(141)
    ax12=fig.add_subplot(142)
    ax21=fig.add_subplot(222)
    ax22=fig.add_subplot(224)
    imh=ax11.imshow(hraster,origin='lower',cmap=fisspy.cm.ha)
    imc=ax12.imshow(craster1,origin='lower',cmap=fisspy.cm.ca)
    
    ax11.set_xlabel('X position [pixel]')
    ax12.set_xlabel('X position [pixel]')
    ax11.set_ylabel('Y position [pixel]')
    ax12.set_ylabel('Y position [pixel]')
    hrmin=hraster.min()
    hrmax=hraster.max()
    crmin=craster.min()
    crmax=craster.max()
    imh.set_clim(hrmin,hrmax)
    imc.set_clim(crmin,crmax)
    
    
    htitle='GST/FISS '+hh['wavelen']+r' $\AA$ '+hh['date']
    ax11.set_title(htitle)
    ctitle='GST/FISS '+ch['wavelen']+r' $\AA$ '+ch['date']
    ax12.set_title(ctitle)
    
    
    hwv=wvcalib(hh)
    cwv=wvcalib(ch)
    x=50
    y=50
    hi=hframe[y,x]
    ci=cframe[y,x]
    p1=ax21.plot(hwv,hi,color='k')
    p2=ax22.plot(cwv,ci,color='k')
    
    ax21.set_title(htitle)
    ax22.set_title(ctitle)
    
    ax21.set_xlabel(r'Wavelength [$\AA$]')
    ax22.set_xlabel(r'Wavelength [$\AA$]')
    ax21.set_ylabel(r'$I_{\lambda}$ [count]')
    ax22.set_ylabel(r'$I_{\lambda}$ [count]')
    ax21.set_xlim(hwv.min(),hwv.max())
    ax22.set_xlim(cwv.min(),cwv.max())
    ax21.set_ylim(hi.min()-100,hi.max()+100)
    ax22.set_ylim(ci.min()-100,ci.max()+100)
    hcur=Cursor(ax11,color='red',useblit=True)
    ccur=Cursor(ax12,color='red',useblit=True)
    hpcur=Cursor(ax21,color='g',useblit=True)
    cpcur=Cursor(ax22,color='g',useblit=True)
    fig.tight_layout()
    fig.canvas.mpl_connect('key_press_event',mark)
    
    root = fig.canvas.manager.window
    panel = QWidget()
    vbox = QVBoxLayout(panel)
    hbox0 = QHBoxLayout(panel)
    hbox = QHBoxLayout(panel)
    hbox2 = QHBoxLayout(panel)
    tsld=Slider('Frame Number(Time)',0,len(hflist)-1,0,value_type='int')
    hmisld=Slider('CamA Color Range Min',500,10000,hrmin,value_type='int')
    hmasld=Slider('CamA Color Range Max',500,10000,hrmax,value_type='int')
    cmisld=Slider('CamB Color Range Min',500,10000,crmin,value_type='int')
    cmasld=Slider('CamB Color Range Max',500,10000,crmax,value_type='int')
    playb=Button('Play (e)',play)
    psb=Button('Pause (space bar)',pauseb)
    
    tsld.slider.valueChanged.connect(time)
    hmisld.slider.valueChanged.connect(colorhrange)
    hmasld.slider.valueChanged.connect(colorhrange)
    cmisld.slider.valueChanged.connect(colorcrange)
    cmasld.slider.valueChanged.connect(colorcrange)
    
    hbox0.addWidget(tsld)
    hbox0.addWidget(playb)
    hbox0.addWidget(psb)
    hbox.addWidget(hmisld)
    hbox.addWidget(hmasld)
    hbox2.addWidget(cmisld)
    hbox2.addWidget(cmasld)
    
    vbox.addLayout(hbox0)
    vbox.addLayout(hbox)
    vbox.addLayout(hbox2)
    panel.setLayout(vbox)

    dock = QDockWidget("Time and Color Range Control (q=mark, w=print, e=play, space-bar=pause/resume g=grid, s=save, h=home, p=move, o=zoom)",root)
    root.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
    dock.setWidget(panel)
    plt.show()

def IFRDDV (lista):
    """
    """
    