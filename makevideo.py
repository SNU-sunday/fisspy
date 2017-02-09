"""
Makevideo

Using the ffmpeg make a video file from images.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from matplotlib.pyplot import imread
import os
import subprocess

__author__="J. Kang: jhkang@astro.snu.ac.kr"
__email__="jhkang@astro.snu.ac.kr"
__date__="Nov 08 2016"

def ffmpeg(imgstr,fpsi,output='video.mp4'):
    """
    FFMPEG
    
    Using the ffmpeg make a video file from images.
    The output video is saved at the same location of images.
    
    Arguments
        imgstr : List of image filename.
        fpsi   : Integer value of Frame Per Second.
        Movie_name : Output video name with extension. Default is video.mp4
    """
    FFMPEG_BIN = "ffmpeg"
    
    exten=output.split('.')[1]
    if exten == 'mp4':
        codec='libx264'
    elif exten == 'avi':
        codec='libxvid'
    elif exten == 'mov':
        codec='mpeg4'
    else:
        codec=''
    
    n=len(imgstr)
    if n == 0:
        raise ValueError('Image list has no element!')
    
    fps=str(fpsi)
    img=imread(imgstr[0])
    size=img.shape
    xsize=size[0]
    ysize=size[1]
    
    if np.mod(xsize*ysize,2) != 0:
        raise ValueError("The size of the image shuld be even numbers.")
    
    newname=np.arange(n)
    newname=np.char.add('_',newname.astype(str))
    newname=np.char.add(newname,'.png')

    dir=os.path.dirname(imgstr[0])
    if bool(dir):
        os.chdir(dir)
    else:
        os.chdir(os.getcwd())

    f=open('img_list.tmp','w')
    for i in imgstr:
        f.write("file '"+os.path.basename(i)+"'\n")
    f.close()
    
    cmd=(FFMPEG_BIN+
         ' -r '+fps+' -f concat -i img_list.tmp'+
         ' -c:v '+codec+' -pix_fmt yuv420p -q:v 1 -y '+output)

    os.system(cmd)
