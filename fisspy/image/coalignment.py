"""
Coalignment

"""
from __future__ import absolute_import, print_function, division

__author__="J. Kang : jhkang@astro.snu.ac.kr"
__date__="Sep 01 2016"

from scipy.fftpack import ifft2,fft2
import numpy as np
from fisspy.io.read import getheader,raster
from . import fiss_sdo_align_tool
from astropy.time import Time
import os
from sunpy.net import vso
from .base import rotation

def alignoffset(image0,template0):
    """
    Alignoffset
    
    Based on the alignoffset.pro IDL code written by Chae 2004
    
    Parameters
    Image : Images for coalignment with the template
            A 2 or 3 Dimensional array ex) image[t,y,x]
    Template : The reference image for coalignment
               2D Dimensional arry ex) template[y,x]
           
    Outputs
        x, y : The single value or array of the offset values.
    ========================================
    Example)
    >>> x, y = alignoffset(image,template)
    ========================================
    
    Notification
    Using for loop is faster than inputing the 3D array as,
    >>> res=np.array([alignoffset(image[i],template) for i in range(nt)])
    where nt is the number of elements for the first axis.
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
    
    cor=ifft2(ifft2(image*gauss)*fft2(template*gauss)).real
    
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
    
    return x, y


def fiss_align_inform(file,wvref=-4,dirname=False,
                      filename=False,save=True,pre_match_wcs=False,
                      sil=True,missing=0):
    """
    fiss_align_infrom
    
    """
    n=len(file)
    hlist=[getheader(i) for i in file]
    tlist=[i['date'] for i in hlist]
    t=Time(tlist,format='isot',scale='ut1')
    dtmin=(t.jd-t.jd[0])*24*60
    
    nx=hlist[0]['naxis3']
    ny=hlist[0]['naxis2']
    
    angle=np.deg2rad(dtmin*0.25)
    
    xc=nx//2
    yc=ny//2
    
    nx1=((nx//2)//2)*2
    ny1=((ny//2)//2)*2
    
    x1=xc-nx1//2
    y1=yc-ny1//2
    
    xa=(x1+np.arange(nx1))
    ya=(y1+np.arange(ny1))[:,None]
    
    im1=raster(file[0],wvref,0.05,x1=x1,x2=x1+nx1,y1=y1,y2=y1+ny1)
    
    dx=np.zeros(n)
    dy=np.zeros(n)
    
    for i in range(n-1):
        #align with next image
        #the alignment is not done with the reference, since the structure can be transformed
        im2=raster(file[i+1],wvref,0.05,x1=x1,x2=x1+nx1-1,y1=y1,y2=y1+ny1-1)
        img1=rotation(im1,angle[i],xa,ya,xc,yc,missing=0)
        img2=rotation(im1,angle[i+1],xa,ya,xc,yc,missing=0)
        sh=alignoffset(img2,img1)
        
        #align with reference
        img1=rotation(im1,angle[i],xa,ya,xc,yc,dx[i],dy[i],missing=0)
        img2=rotation(im2,angle[i+1],xa,ya,xc,yc,dx[i]+sh[0],dx[i]+sh[1],missing=0)
        sh+=alignoffset(img2,img1)
        
        dx[i+1]=dx[i]+sh[0]
        dy[i+1]=dy[i]+sh[1]
        
        im1=im2
        
        if not sil:
            print(i)
    
    result=dict(xc=xc,yc=yc,angle=angle,dt=dtmin,dx=dx,dy=dy)
    if save:
        if not dirname:
            dirname=os.getcwd()+os.sep
        if not filename:
            filename=t[0].value[:10]
        filename2=dirname+filename
        if not pre_match_wcs:
            np.savez(filename2+'_align_lev0.npz',xc=xc,yc=yc,angle=angle,
                     dt=dtmin,dx=dx,dy=dy)
        else:
            tmp=np.load(filename2+'_match_wcs.npz')
            np.savez(filename2+'_align_lev1.npz',xc=xc,yc=yc,angle=angle,
                     dt=dtmin,dx=dx,dy=dy,sdo_angle=tmp['sdo_angle'],
                     wcsx=tmp['wcsx'],wcsy=tmp['wcsy'])
            os.remove(filename2+'_match_wcs.npz')
    return result
    

    
def match_wcs(fiss_file,sdo_file=False,dirname=False,
              filename=False,sil=True,sdo_path=False,
              manual=True,wvref=-4,reflect=True,alpha=0.5):
    """
    match_wcs
    """
    
    if not sdo_file:
        h=getheader(fiss_file)
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
        
        if not sdo_path:
            sdo_path=os.getcwd()+os.sep()
        sdo_file=(vc.get(res,path=sdo_path+'{file}',methods=('URL-FILE','URL')).wait())[0]
    
    fiss_sdo_align_tool.manual(fiss_file,sdo_file,dirname=dirname,
                               filename=filename,wvref=wvref,
                               reflect=reflect,alpha=alpha)
    return
