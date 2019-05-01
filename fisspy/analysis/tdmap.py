"""
"""

from __future__ import absolute_import, division
import numpy as np
from interpolation.splines import LinearSpline

__author__= "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

def makeTDmap(arr, R, theta, extent=[0, 'end', 0, 'end'], xc=0, yc=0):
    """
    arr: (t, y, x) 3D array
    R : radius
    theta: angle
    """
    
    angle = np.deg2rad(theta)
    nt, ny, nx = arr.shape
    if extent[1] == 'end':
        extent[1] = ny-1
        extent[3] = nx-1
    smin = [extent[0], extent[2]]
    smax = [extent[1], extent[3]]
    order = [ny, nx]
    x1 = -R*np.cos(angle)+xc
    x2 = R*np.cos(angle)+xc
    y1 = -R*np.sin(angle)+xc
    y2 = R*np.sin(angle)+xc
    nl = int(np.ceil(2*R/(extent[1]-extent[0])*nx))
    x = np.linspace(x1, x2, nl)
    y = np.linspace(y1, y2, nl)
    
    td = np.empty([nl, nt])
    for i, ta in enumerate(arr):
        interp = LinearSpline(smin, smax, order, ta)
        iarr = np.array([y,x]).T
        td[:,i] = interp(iarr).T
        
    return td