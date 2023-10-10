"""
Read the FISS fts file and its header.

"""
from __future__ import absolute_import, division

__author__ = "Juhyeong Kang"
__email__ = "jhkang@astro.snu.ac.kr"

from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve as conv
import numpy as np
import os

__all__ = ['frame', 'pca_read', 'raster', 'getheader', 'frame2raster',
           'sp_av', 'sp_med', 'wavecalib', 'simple_wvcalib']

def frame(file, x1=0, x2=False, pca=True, ncoeff=False, xmax=False,
          smooth=False, **kwargs):
    """Read the FISS fts file.

    Parameters
    ----------
    file : str
        A string of file name to be read.
    x1 : int
        A starting index of the frame along the scanning direction.
    x2 : (optional) int
        An ending index of the frame along the scanning direction.
        If not, then the only x1 frame is read.
    pca : (optional) bool
        If True, the frame is read from the PCA file.
        Default is True, but the function automatically check
        the existance of the pca file.
    ncoeff : (optional) int
        The number of coefficients to be used for
        the construction of frame in a pca file.
    xmax : (optional) bool
        If True, the x2 value is set as the maximum end point of the frame.
            * Default is False.
    smooth : (optional) bool
        If True, apply the Savitzky-Golay filter to increase the signal to
        noise without greatly distorting the signal of the given fts file.
            * Default is False.
    nsmooth : (optional) int
        The number of smooting.
        Default is 1 for the case of the compressed file,
        and is 2 for the case of the uncompresseed file.
    kwargs 
        The parameters for smooth (savitzky-golay filter), \n
        See the docstring of the `scipy.signal.savgol_filter`.
        
    Returns
    -------
    frame : ~numpy.ndarray
        FISS data frame with the information of (wavelength, y, x).

    References
    ----------
    `Savitzky-Golay filter <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_.\n
    `scipy.signal.savgol_filter <https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter>`_
    
    Notes
    -----
        This function is based on the IDL code FISS_READ_FRAME.PRO 
        written by J. Chae, 2013.
    
        This function automatically check the existance of the pca
        file by reading the fts header.
        
    Example
    -------
    .. plot::
        :include-source:
            
        import matplotlib.pyplot as plt
        from fisspy.io import read
        import fisspy
        import fisspy.data.sample
        data=read.frame(fisspy.data.sample.FISS_IMAGE,xmax=True)
        plt.imshow(data[:,75],cmap=fisspy.cm.ca,origin='lower',interpolation='bilinear')
        plt.title(r"GST/FISS 8542 $\AA$ Spectrogram")
        plt.show()
    """
    if not file:
        raise ValueError('Empty filename')
    if x2 and x2 <= x1:
        raise ValueError('x2 must be larger than x1')
    
    header=fits.getheader(file)
    
    try:
        header['pfile']
    except:
        pca=False
    
    if xmax and not x2:
        x2=header['naxis3']
    elif not x2:
        x2=x1+1
    
    if pca:
        spec=pca_read(file,header,x1,x2,ncoeff=ncoeff)
    else:        
        spec=fits.getdata(file)[x1:x2]
    if x1+1 == x2:
        spec=spec[0]
        return spec
    spec=spec.transpose((1,0,2)).astype(float)
    
    if smooth:
        winl=kwargs.pop('window_length',7)
        pord=kwargs.pop('polyorder',3)
        deriv=kwargs.pop('deriv',0)
        delta=kwargs.pop('delta',1.0)
        mode=kwargs.pop('mode','interp')
        cval=kwargs.pop('cval',0.0)
        nsmooth=kwargs.pop('nsmooth',int(not pca)+1)
            
        for i in range(nsmooth):
            spec=savgol_filter(spec,winl,pord,deriv=deriv,
                               delta=delta,mode=mode,cval=cval)
    return spec


def pca_read(file,header,x1,x2=False,ncoeff=False):
    """
    Read the pca compressed FISS fts file.
    
    Parameters
    ----------
    file : str
        A string of file name to be read.
    header : astropy.io.fits.header.Header
        The fts file header.
    x1   : int
        A starting index of the frame along the scanning direction.
    x2   : (optional) int
        An ending index of the frame along the scanning direction.
        If not, then the only x1 frame is read.
    ncoeff : (optional) int
        The number of coefficients to be used for
        the construction of frame in a pca file.
    
    Returns
    -------
    frame : ~numpy.ndarry
        FISS data frame with the information of (wavelength, y, x).
        
    Notes
    -----
        This function is based on the IDL code FISS_PCA_READ.PRO
        written by J. Chae, 2013.
        The required fts data are two. One is the "_c.fts",
        and the other is "_p.fts"
    
    """
    if not file:
        raise ValueError('Empty filename')
    if not x2:
        x2 = x1+1
        
    dir = os.path.dirname(file)
    pfile = header['pfile']
    
    if dir:
        pfile = os.path.join(dir, pfile)
        
    pdata = fits.getdata(pfile)
    data = fits.getdata(file)[x1:x2]
    ncoeff1 = data.shape[2]-1
    if not ncoeff:
        ncoeff = ncoeff1
    elif ncoeff > ncoeff1:
        ncoeff = ncoeff1
    
    spec = np.dot(data[:,:,0:ncoeff],pdata[0:ncoeff,:])
    spec *= 10.**data[:,:,ncoeff][:,:,None]
    return spec

def raster(file, wv, hw=0.05, x1=0, x2=False, y1=0, y2=False,
           pca=True, smooth=False, absScale = False, **kwargs):
    """
    Make raster images for a given file at wv of wavelength within width hw
    
    Parameters
    ----------
    file : str
        A string of file name to be read.
    wv   : float or 1d ndarray
        Referenced wavelengths.
    hw   : float
        A half-width of wavelength integration in unit of Angstrom.
        Default is 0.05
    x1   : (optional) int
        A starting index of the frame along the scanning direction.
    x2   : (optional) int
        An ending index of the frame along the scanning direction.
        If not, x2 is set to the maximum end point of the frame.
    y1   : (optional) int
        A starting index of the frame along the slit position.
    y2   : (optional0 int
        A ending index of the frame along the slit position.
    pca  : (optional) bool
        If True, the frame is read from the PCA file.
        Default is True, but the function automatically check
        the existance of the pca file.
    absScale : (optional) bool
        If True, the wavelength should be given in absolute scale.
        If Flase, the wavelength should be given in relative scale. 
    smooth : (optional) bool
        If True, apply the Savitzky-Golay filter to increase the signal to
        noise without greatly distorting the signal of the given fts file.
            * Default is False.
    kwargs
        Any additional keyword arguments to read frame.
        See the docstring of `fisspy.io.read.frame`
        
    Returns
    -------
    Raster : ~numpy.ndarray
        Raster image at given wavelengths.
        
    Notes
    -----
        This function is based on the IDL code FISS_RASTER.PRO
        written by J. Chae, 2013.
        This function automatically check the existance of the pca file by
        reading the fts header.
    
    Example
    -------
    .. plot::
        :include-source:
            
        import matplotlib.pyplot as plt
        from fisspy.io import read
        from fisspy import cm
        import fisspy.data.sample
        raster=read.raster(fisspy.data.sample.FISS_IMAGE,0.3)
        plt.imshow(raster, cmap=cm.ca, origin='lower', interpolation='bilinear')
        plt.title(r"GST/FISS 8542+0.3 $\AA$ Spectrogram")
        plt.show()
    """
    header = getheader(file,pca)
    ny = header['NAXIS2']
    nx = header['NAXIS3']
    dldw = header['CDELT1']
    
    if not file:
        raise ValueError('Empty filename')
    if x2 and x2 <= x1+1:
        raise ValueError('x2 must be larger than x1+1')
    
    try:
        num = wv.shape[0]    
    except:
        num = 1
        wv = np.array([wv])
    
    if not x2:
        x2 = int(nx)
    if not y2:
        y2 = int(ny)
    
    wl = simple_wvcalib(header, absScale= absScale)
    if hw < abs(dldw)/2.:
        hw = abs(dldw)/2.
    
    s = np.abs(wl-wv[:,None])<=hw
    sp = frame(file,x1,x2,pca=pca,smooth=smooth,**kwargs)
    leng = s.sum(1)
    if num == 1:
        img = sp[y1:y2,:,s[0,:]].sum(2)/leng[0]
        return img.reshape((y2-y1,x2-x1))
    else:
        img=np.array([sp[y1:y2,:,s[i,:]].sum(2)/leng[i] for i in range(num)])
        return img.reshape((num,y2-y1,x2-x1))


def getheader(file,pca=True):
    """
    Get the FISS fts file header.
    
    Parameters
    ----------
    file : str
        A string of file name to be read.
    pca  : (optional) bool
        If True, the frame is read from the PCA file.
        Default is True, but the function automatically check
        the existance of the pca file.
    
    Returns
    -------
    header : astropy.io.fits.Header
        The fts file header.
    
    Notes
    -----
        This function automatically check the existance of the pca file by
        reading the fts header.
    
    Example
    -------
    >>> from fisspy.io import read
    >>> h=read.getheader(file)
    >>> h['date']
    '2014-06-03T16:49:42'
    """
    header0 = fits.getheader(file)
    
    pfile = header0.pop('pfile',False)
    if not pfile:
        return header0
        
    header = fits.Header()
    if pca:
        header['pfile']=pfile
        for i in header0['comment']:
            sori = i.split('=')
            if len(sori) == 1:
                skv = sori[0].split(None,1)
                if len(skv) == 1:
                    pass
                else:
                    header[skv[0]] = skv[1]
            else:
                key = sori[0]
                svc = sori[1].split('/')
                try:
                    item = float(svc[0])
                except:
                    item = svc[0].split("'")
                    if len(item) != 1:
                        item = item[1].split(None,0)[0]
                    else:
                        item = item[0].split(None,0)[0]
                try:
                    if item-int(svc[0]) == 0:
                        item = int(item)
                except:
                    pass
                if len(svc) == 1:
                    header[key] = item
                else:
                    header[key] = (item,svc[1])
                    
    header['simple'] = True
    alignl=header0.pop('alignl',-1)
    
    if alignl == 0:
        keys=['reflect','reffr','reffi','cdelt2','cdelt3','crota2',
              'crpix3','shift3','crpix2','shift2','margin2','margin3']
        header['alignl'] = (alignl,'Alignment level')
        for i in keys:
            header[i] = (header0[i],header0.comments[i])
        header['history'] = str(header0['history'])
    if alignl == 1:
        keys=['reflect','reffr','reffi','cdelt2','cdelt3','crota1',
              'crota2','crpix3','crval3','shift3','crpix2','crval2',
              'shift2','margin2','margin3']
        header['alignl'] = (alignl,'Alignment level')
        for i in keys:
            header[i] = (header0[i],header0.comments[i])
        header['history'] = str(header0['history'])
        
    return header

def frame2raster(frame, header, wv, absScale = False):
    """
    Make a raster image by using the frame data.
    
    Parameters
    ----------
    frame : ~numpy.ndarray
        Data which is read from the fisspy.io.read.frame
    header : astropy.io.fits.Header
        FISS data header
    wv : float or ~numpy.ndarray
        Referenced wavelengths to draw raster image. It must be the one single float,
        or 1D array
    absScale : (optional) bool
        If True, the wavelength should be given in absolute scale.
        If Flase, the wavelength should be given in relative scale. 
    Returns
    -------
    Raster : ~numpy.ndarray
        Raster image at gieven wavelength.
    """
    hw = 0.05
    wl = simple_wvcalib(header, absScale= absScale)
    s = np.abs(wl - wv) <= hw
    img = frame[:, :, s].sum(2) / s.sum()
    return img

def sp_av(file) :
    a = frame(file, xmax = True)
    return a.mean(axis = 1)

def sp_med(file) :
    a = frame(file, xmax = True)
    return np.median(a, axis = 1)

def wavecalib(band,profile,method=True):
    """
    Calibrate the wavelength for FISS spectrum profile.
    
    Parameters
    ----------
    band : str
        A string to identify the wavelength.
        Allowable wavelength bands are '6562','8542','5890','5434'
    profile : ~numpy.ndarray
        A 1 dimensional numpy array of spectral profile.
    Method : (optional) bool
        * Default is True.
        If true, the reference lines for calibration are the telluric lines.
        Else if False, the reference lines are the solar absorption lines.
    
    Returns
    -------
    wavelength : ~numpy.ndarray
        Calibrated wavelength.
    
    Notes
    -----
        This function is based on the FISS IDL code FISS_WV_CALIB.PRO
        written by J. Chae, 2013.
    
    Example
    -------
    >>> from fisspy.analysis import doppler
    >>> wv=doppler.wavecalib('6562',profile)
    
    """
    band=band[0:4]
    nw=profile.shape[0]
    
    if method:
        if band == '6562':
            line=np.array([6561.097,6564.206])
            lamb0=6562.817
            dldw=0.019182
        elif band == '8542':
            line=np.array([8540.817,8546.222])
            lamb0=8542.090
            dldw=-0.026252
        elif band == '5890':
            line=np.array([5889.951,5892.898])
            lamb0=5889.9509
            dldw=0.016847
        elif band == '5434':
            line=np.array([5434.524,5436.596])
            lamb0=5434.5235
            dldw=-0.016847
        else:
            raise ValueError("The wavelength band value is not allowable.\n"+
                             "Please select the wavelenth "+
                             "among '6562','8542','5890','5434'")
    else:
        if band == '6562':
            line=np.array([6562.817,6559.580])
            lamb0=6562.817
            dldw=0.019182
        elif band == '8542':
            line=np.array([8542.089,8537.930])
            lamb0=8542.090
            dldw=-0.026252
        else:
            raise ValueError("The wavelength band value is not allowable.\n"
                             "Please select the wavelenth "
                             "among '6562','8542','5890','5434'")
    
    w=np.arange(nw)
    wl=np.zeros(2)
    wc=profile[20:nw-20].argmin()+20
    lamb=(w-wc)*dldw+lamb0
    
    for i in range(2):
        mask=np.abs(lamb-line[i]) <= 0.3
        wtmp=w[mask]
        ptmp=conv(profile[mask],[-1,2,-1],'same')
        mask2=ptmp[1:-1].argmin()+1
        try:
            wtmp=wtmp[mask2-3:mask2+4]
            ptmp=ptmp[mask2-3:mask2+4]
        except:
            raise ValueError('Fail to wavelength calibration\n'
            'please change the method %s to %s' %(repr(method), repr(not method)))
        c=np.polyfit(wtmp-np.median(wtmp),ptmp,2)
        wl[i]=np.median(wtmp)-c[1]/(2*c[0])    #local minimum of the profile
    
    dldw=(line[1]-line[0])/(wl[1]-wl[0])
    wc=wl[0]-(line[0]-lamb0)/dldw
    wavelength=(w-wc)*dldw
    
    return wavelength

def simple_wvcalib(header, absScale = False):
    if absScale:
        return (np.arange(header['naxis1']) -
                header['crpix1'])*header['cdelt1'] + header['crval1']
    else:
        return (np.arange(header['naxis1']) -
                header['crpix1'])*header['cdelt1']