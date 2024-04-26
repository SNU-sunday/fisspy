from .base import alignOffset, rotImage, shiftImage
import numpy as np
from astropy.time import Time
from ..read import FISS
from os.path import join
from os import getcwd

__author__ = "Juhyung Kang"
__email__ = "jhkang0301@gmail.com"
__all__= ["calAlignPars", "alignCams", "writeAlignPars", "readAlignPars", "alignAll", "alignDataCube", 'alignTwoDataCubes', 'saveAlignCube', 'makeExample']

def calAlignPars(lfiles, refFrame=None):
    """
    Calculate the parameters to be used for the alignment for given series of the line spectra.

    Parameters
    ----------
    lfiles: `list`
        A series of the FISS data files (either cam A or cam B)
    refFrame: `int` (optional)
        Reference frame number.
        If None, the middle frame (time) is considered as a reference.
        Default is None.

    Returns
    -------
    alinPars: `dict`
        Parameter to align the data files.
        It consists of 9 elements:
            - cam: camera information either 'A' or 'B'
            - refFrame: Reference frame number.
            - refTime: (str) isot time for the reference frame. 
            - time: relative time to the reference time in the unit of second.
            - xc: x posiotion of the center of the rotation.,
            - yc: y posiotion of the center of the rotation.,
            - dx: shift in the direction of the x axis.
            - dy: shift in the direction of the y axis.
            - angle: angle of the image rotation.
    """
    nf = len(lfiles)
    if nf <= 1:
        raise ValueError(f"The number of elements of lfiles should be larger than 1.\n    Note) nf={nf}.")
    if refFrame is None:
        rf = nf//2
    else:
        rf = refFrame
    lfiles.sort()
    fissr = FISS(lfiles[rf], wvCalibMethod='simple')
    imr = fissr.data[:,::-1,50:55].mean(2)
    refTime = fissr.date
    rt = Time(refTime).jd
    time = np.zeros(nf, dtype=float)
    xc = np.zeros(nf, dtype=float)
    yc = np.zeros(nf, dtype=float)
    dx = np.zeros(nf, dtype=float)
    dy = np.zeros(nf, dtype=float)
    angle = np.zeros(nf, dtype=float)
    
    im0 = imr

    print('Running Alignment')
    print('    0%', end='\r', flush=True)
    
    for k in range(nf):
        print(f'    {(k+1)*100/nf}%', end='\r', flush=True)
        if k < rf:
            # backward alignment
            i0 = rf - k
            i = rf - (k + 1)
        elif k == rf:
            im0 = imr
            time[k] = 0
            xc[k] = fissr.nx/2
            yc[k] = fissr.ny/2
            continue
        else:
            # forward alignment
            i0 = k-1
            i = k
        fiss = FISS(lfiles[i])
        im = fiss.data[:,::-1,50:55].mean(2)
        t = Time(fiss.date).jd - rt
        xc[i] = fiss.nx/2
        yc[i] = fiss.ny/2
        time[i] = t*24*3600
        angle[i] = -t*(2*np.pi)
        dangle = angle[i] - angle[i0]
        rim = rotImage(im, dangle, missing=None)
        for rep in range(2):
            sh = alignOffset(rim, im0)
            rim = shiftImage(rim, -sh, missing=None)
            dx[i] += sh[1]
            dy[i] += sh[0]
        dx[i] += dx[i0]
        dy[i] += dy[i0]
        im0 = im

    print('Done        ')

    alignPars = {
              "cam": fissr.cam,
              "refFrame": rf,
              "refTime": refTime,
              "time": time,
              "xc": xc,
              "yc": yc,
              "dx": dx,
              "dy": dy,
              "angle": angle
              }
    
    return alignPars

def alignCams(frefA, frefB, refCam='A'):
    """
    Align two cameras.

    Parameters
    ----------
    frefA: `str`
        Filename of the camera A for the reference frame.
    frefB: `str`
        Filename of the camera B for the reference frame.
    refCam: `str` (optional)
        Reference camera for the alignment.
        Either 'A' or 'B'
        Default is 'A'

    Returns
    -------
    refCam: `str`
        Reference camera for the alignment.
    dx: `float`
        Shift in the direction of the x axis
    dy: `float`
        Shift in the direction of the y axis
    """
    rc = refCam.upper()
    if rc != 'A' and rc != 'B':
        raise ValueError("refCam should be either 'A' or 'B'.")
    fB = FISS(frefB)
    nx = fB.nx
    ny = fB.ny
    fA = FISS(frefA, x2=nx, y2=ny)
    
    imA = fA.data[:,::-1,50:55].mean(2)
    imB = fB.data[:,::-1,50:55].mean(2)

    if rc == 'A':
        imr = imA
        im = imB
    else:
        imr = imB
        im = imA

    dx = 0
    dy = 0
    for rep in range(3):
        sh = alignOffset(im, imr)
        im = shiftImage(im, -sh, missing=None)
        dx += sh[1]
        dy += sh[0]

    return dx, dy
    
def writeAlignPars(apar, refCam=None, sname=None):
    """
    Write file for the align parameters.

    Parameters
    ----------
    apar: `dict`
        Parameter to align the data files.
        See `~fisspy.align.alignment.calAlignPars`.
    refCam: `str` (optional)
        Reference camera for the alignment.
    sname: `str` (optional)
        Save file name.
        The extension should be .npz
        Default is alignpar_{apar['cam']} in the current working directory.

    Returns
    -------
    None
    """
    if sname is None:
        fname = join(getcwd(), f"alignpar_{apar['cam']}.npz")
    else:
        fname = sname
        if fname.split('.')[-1] != 'npz':
            fname = fname+'.npz'
        
    print(f'Write alignpar: {fname} .')
    np.savez(fname,
             cam=apar['cam'],
             refFrame=apar['refFrame'],
             refTime=apar['refTime'],
             time=apar['time'],
             xc=apar['xc'],
             yc=apar['yc'],
             dx=apar['dx'],
             dy=apar['dy'],
             angle=apar['angle'],
             refCam=refCam)
    
def readAlignPars(apfile):
    """
    Read file for the align parameters.

    Parameters
    ----------
    apfile: `str`
        Filename for the align parameters.

    Returns
    -------
    apar: `~numpy.lib.npyio.NpzFile`
        align parameters.
            -keys: ['cam', 'refFrame', 'refTime', 'time', 'xc', 'yc', 'dx', 'dy', 'angle', 'refCam']
    """
    return np.load(apfile)

def alignAll(lfA, lfB, refFrame=None, refCam='A', sname=None, save=True):
    """
    Parameters
    ----------
    lfA: `list`
        A series of the camA data files.
    lfB: `list`
        A series of the camB data files.
    refFrame: `int` (optional)
        Reference frame number.
        If None, the middle frame (time) is considered as a reference.
        Default is None.
    refCam: `str` (optional)
        Reference camera for the alignment.
        Either 'A' or 'B'
        Default is 'A'
    save: `bool` (optional)
        Save align paramereters in the working directory.
        Default is True.

    Returns
    -------
    alignParsA: `dict`
        alignPars for cam A.
        See `~fisspy.align.alignment.calAlignPars`.
    alignParsB: `dict`
        alignPars for cam B.
        See `~fisspy.align.alignment.calAlignPars`.
    """
    if len(lfA) != len(lfB):
        raise ValueError("The size of two lists of lfA and lfB should be the same.")
    
    rc = refCam.upper()
    if rc != 'A' and rc != 'B':
        raise ValueError("refCam should be either 'A' or 'B'.")
    
    print('Align cam A.')
    aparA = calAlignPars(lfA, refFrame=refFrame)

    print('Align cam B.')
    aparB = calAlignPars(lfB, refFrame=refFrame)

    rf = aparA['refFrame']
    print("Align two cameras")
    dx, dy = alignCams(lfA[rf], lfB[rf], refCam=refCam)

    if refCam == 'A':
        aparB['dx'] += dx
        aparB['dy'] += dy
    else:
        aparA['dx'] += dx
        aparA['dy'] += dy

    if save:
        if sname is None:
            snameA = None
            snameB = None
        else:
            sp = sname.split('.npz')
            
            if sp[0][-1] == 'A':
                snameA = sp[0]
                snameB = sp[0][:-1]+'B'
            elif sp[0][-1] == 'B':
                snameA = sp[0][:-1]+'A'
                snameB = sp[0]
            else:
                snameA = sp[0]+'A'
                snameB = sp[0]+'B'
            
            snameA = snameA + '.npz'
            snameB = snameB + '.npz'

        writeAlignPars(aparA, refCam=refCam, sname=snameA)
        writeAlignPars(aparB, refCam=refCam, sname=snameB)

    return aparA, aparB

def alignDataCube(data, fapar, xmargin=None, ymargin=None, cubic=False):
    """
    Align 3D data cube for given apar.
    Note that the data will be flip in the x axis to correct the mirror reversal.
    Please do not use this function when you use two data cubes of two cams, but use `~fisspy.align.alignment.alignTwoDataCubes`.

    Parameters
    ----------
    data: `~numpy.ndarray`
        3-dimensional data array with the shape of (nt, ny, nx).
    fapar: `dict`
        File name of the alignpar.
    xmargin: `int`
        Margin for x-axis.
        The size of the x-axis increases to nx + 2*xmargin.
        If None, automatically calculate the margin.
        Default is None.
    ymargin: `int`
        Margin for y-axis
        The size of the x-axis increases to ny + 2*ymargin.
        If None, automatically calculate the margin.
        Default is None.
    cubic: `bool`, (optional)
        Use cubic interpolation to determine the value in the aligned position.
        If False, use linear interpolation.
        Default is None.
    
    Returns
    -------
    cdata: `~numpy.ndarray`
        Aligned data.
    """
    if data.ndim != 3:
        raise ValueError("Dimension of the Data should be 3.")
    nt, ny, nx = data.shape
    apar = readAlignPars(fapar)
    time = apar['time']
    nap = len(time)
    if nt != nap:
        raise ValueError("Array size is different from the size of the apar.")
    
    xc = apar['xc']
    yc = apar['yc']
    ang = apar['angle']

    # margin
    l = ((nx//2)**2+(ny//2)**2)**0.5
    ang0 = np.arctan2(ny//2,nx//2)
    aa = apar['angle']+ang0
    if xmargin is None:
        xm = int(l*np.cos(aa).max() - nx//2 + 0.5)
    else:
        xm = xmargin
    if ymargin is None:
        ym = int(l*np.sin(aa).max() - ny//2 + 0.5)
    else:
        ym = ymargin
    
    cdata = np.zeros((nt, ny+2*ym, nx+2*xm),dtype='float')

    for i, d in enumerate(data):
        rimg = rotImage(d[:,::-1], ang[i],
                        xc=xc[i], yc=yc[i],
                        dx=apar['dx'][i],
                        dy=apar['dy'][i],
                        xmargin=xm, ymargin=ym,
                        cubic=cubic)
        cdata[i] = rimg

    return cdata

def alignTwoDataCubes(dataA, dataB, faparA, faparB, xmargin=None, ymargin=None, cubic=False):
    """
    Align two 3D data cubes.
    Note that the data will be flip in the x axis to correct the mirror reversal.

    Parameters
    ----------
    dataA: `~numpy.ndarray`
        3-dimensional data array for cam A with the shape of (nt, ny, nx).
    dataB: `~numpy.ndarray`
        3-dimensional data array for cam B with the shape of (nt, ny, nx).
    faparA: `str`
        File name of the alignpar for cam A.
    faparB: `str`
        File name of the alignpar for cam B.
    xmargin: `int`, (optional)
        Margin for x-axis.
        The size of the x-axis increases to nx + 2*xmargin.
        If None, automatically calculate the margin.
        Default is None.
    ymargin: `int`, (optional)
        Margin for y-axis
        The size of the x-axis increases to ny + 2*ymargin.
        If None, automatically calculate the margin.
        Default is None.
    cubic: `bool`, (optional)
        Use cubic interpolation to determine the value in the aligned position.
        If False, use linear interpolation.
        Default is None.
    
    Returns
    -------
    cdata: `~numpy.ndarray`
        Aligned data.
    """
    nt, ny, nx = dataA.shape
    l = ((nx//2)**2+(ny//2)**2)**0.5
    aparA = readAlignPars(faparA)
    aparB = readAlignPars(faparB)
    ang0 = np.arctan2(ny//2,nx//2)
    ang = aparA['angle']+ang0
    if xmargin is None:
        xm = int(l*np.cos(ang).max() - nx//2 + int(abs(aparB['dx'][100])) + 0.5)
    else:
        xm = xmargin
    if ymargin is None:
        ym = int(l*np.sin(ang).max() - ny//2 + int(abs(aparB['dy'][100])) + 0.5)
    else:
        ym = ymargin

    cdataA = alignDataCube(dataA, faparA, xmargin=xm, ymargin=ym, cubic=cubic)
    cdataB = alignDataCube(dataB, faparB, xmargin=xm, ymargin=ym, cubic=cubic)
    return cdataA, cdataB

def saveAlignCube(adata, time, sname, dt=None, dx=0.16*725, dy=0.16*725):
    """
    Save aligned data cube.
    
    Parameters
    ----------
    adata: `~numpy.ndarray`
        3D array of the aligned data (nt,ny,nx).
    time: `~numpy.ndarray`
        1D time array in unit of second.
    sname: `str`
        filename to save.
        Extension should be .npz.
    dt: `float` (optional)
        Pixel scale along the t-axis in unit of second.
    dx: `float` (optional)
        Pixel scale along the x-axis in unit of km.
    dy: `float` (optional)
        Pixel scale along the y-axis in unit of km.

    Returns
    -------
    None
    """
    sp = sname.split('.npz')
    if len(sp) <= 1:
        raise ValueError("Extension should be .npz.")
    
    if dt is None:
        t = np.roll(time, -1) - time
        dt1 = np.median(t[:-1])
    else:
        dt1 = dt
        
    np.savez(sname, data=adata, time=time, dx=dx, dy=dy, dt=dt1)

def makeExample(lfA, lfB, faparA, faparB):
    """
    Make aligned data images as an example.

    Parameters
    ----------
    lfA: `list`
        A series of the cam A data files.
    lfB: `list`
        A series of the cam B data files.
    faparA: `str`
        Filename of the align parameter for the cam A.
    faparB: `str`
        Filename of the align parameter for the cam B.
    
    Returns
    -------
    adataA: `~numpy.ndarray`
        Aligned data cube for the cam A.
    adataB: `~numpy.ndarray`
        Aligned data cube for the cam B.
    """
    nt = len(lfA)
    fissr = FISS(lfA[nt//2])
    nx = fissr.nx
    ny = fissr.ny
    fissrB = FISS(lfB[nt//2])
    nxB = fissrB.nx
    nyB = fissrB.ny
    dataA = np.zeros((nt, ny, nx), dtype=float)
    dataB = np.zeros((nt, nyB, nxB), dtype=float)

    print('Running make cube')
    print('    0%', end='\r', flush=True)

    for i, f in enumerate(lfA):
        print(f'    {(i+1)*100/nt}%', end='\r', flush=True)
        fiss = FISS(f)
        fissB = FISS(lfB[i])
        dataA[i] = fiss.data[...,50:55].mean(2)
        dataB[i] = fissB.data[...,50:55].mean(2)

    print('Done             ')
    apar = readAlignPars(faparA)
    aparB = readAlignPars(faparB)
    # l = ((nx//2)**2+(ny//2)**2)**0.5
    # ang0 = np.arctan2(ny//2,nx//2)
    # ang = apar['angle']+ang0
    # xm = int(l*np.cos(ang).max() - nx//2 + int(abs(aparB['dx'][100])) + 0.5)
    # ym = int(l*np.sin(ang).max() - ny//2 + int(abs(aparB['dy'][100])) + 0.5)
    # print(f"xmargin:{xm}, ymargin:{ym}")
    # adataA = alignDataCube(dataA, faparA, xmargin=xm, ymargin=ym)
    # adataB = alignDataCube(dataB, faparB, xmargin=xm, ymargin=ym)
    adataA, adataB = alignTwoDataCubes(dataA, dataB, faparA, faparB)

    return adataA, adataB