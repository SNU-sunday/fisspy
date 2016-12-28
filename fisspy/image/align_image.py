"""
Make the aligned fiss image.

"""
from __future__ import absolute_import, division

import numpy as np
from fisspy.image import base

__author__="Juhyeong Kang"
__email__="jhkang@astro.snu.ac.kr"

def fiss_align_cube(fiss_img,alignfile,ref_frame=0,
                    reflection=True,margin=True,level=False):
    """
    
    """
    if not level:
        level=alignfile[-8:-4]
    
    inform=np.load(alignfile)
    dt=inform['dt']
    angle=inform['angle']
    dx=inform['dx']
    dy=inform['dy']
    xc=inform['xc']
    yc=inform['yc']
    n=len(dt)
    
    if ref_frame != 0:
        dx-=dx[ref_frame]
        dy-=dy[ref_frame]
    
    nt,ny,nx=fiss_img.shape
    
    if nt != n:
        raise ValueError('The number of image is not equal to the size of'\
                         'align information')
        
    if level=='lev0':
        angle-=angle[ref_frame]
        x=np.array((0,nx-1,nx-1,0))
        y=np.array((0,0,ny-1,ny-1))
        xt1,yt1=base.rot_trans(x,y,xc,yc,angle.max())
        xt2,yt2=base.rot_trans(x,y,xc,yc,angle.min())
        tmpx=np.concatenate((xt1,xt2))
        tmpy=np.concatenate((yt1,yt2))
        xmargin=np.abs(np.round(tmpx.min()+dx.min()))+1
        ymargin=np.abs(np.round(tmpy.min()+dy.min()))+1
        nx1=int(nx+2*xmargin)
        ny1=int(ny+2*ymargin)
        
        imgout=np.zeros((nt,ny1,nx1))
        for i in range(nt):
            imgout[i]=base.rot(fiss_img[i],angle[i],xc,yc,dx[i],dy[i],
                                    xmargin,ymargin,missing=0)
    elif level=='lev1':
        sdo_angle=np.deg2rad(inform['sdo_angle'])
        if reflection:
            fiss_img=fiss_img.T
            angle=sdo_angle-angle
            x=np.array((0,ny-1,ny-1,0))
            y=np.array((0,0,nx-1,nx-1))
            xt1,yt1=base.rot_trans(x,y,yc,xc,angle.max())
            xt2,yt2=base.rot_trans(x,y,yc,xc,angle.min())
            tmpx=np.concatenate((xt1,xt2))
            tmpy=np.concatenate((yt1,yt2))
            xmargin=np.abs(np.round(tmpx.min()+dy.min()))+1
            ymargin=np.abs(np.round(tmpy.min()+dx.min()))+1
            nx1=int(ny+2*xmargin)
            ny1=int(nx+2*ymargin)
            
            imgout=np.zeros((nt,ny1,nx1))
            for i in range(nt):
                imgout[i]=base.rot(fiss_img[:,:,i],angle[i],yc,xc,
                                        dy[i],dx[i],xmargin,ymargin,missing=0)
        else:
            angle=sdo_angle+angle
            xt1,yt1=base.rot_trans(x,y,xc,yc,angle.max())
            xt2,yt2=base.rot_trans(x,y,xc,yc,angle.min())
            tmpx=np.concatenate((xt1,xt2))
            tmpy=np.concatenate((yt1,yt2))
            xmargin=np.abs(np.round(tmpx.min()+dx.min()))
            ymargin=np.abs(np.round(tmpy.min()+dy.min()))
            nx1=int(nx+2*xmargin)
            ny1=int(ny+2*ymargin)
            
            
            imgout=np.zeros((nt,ny1,nx1))
            for i in range(nt):
                imgout[i]=base.rot(fiss_img[i],angle[i],xc,yc,
                                        dx[i],dy[i],xmargin,ymargin,missing=0)
    else:
        raise ValueError('Can not be identify the level of alignfile')
    if margin:
        return imgout, xmargin, ymargin
    else:
        return imgout