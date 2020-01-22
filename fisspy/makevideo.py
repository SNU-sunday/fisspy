"""
Makevideo

Using the ffmpeg make a video file from images.

HISTORY
-------
* Nov 08 2016: Make a video by using ffmpeg program
* Jan 22 2020: Make a video by using matplotlib animation module
"""
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname, basename, join

from PIL import Image
import warnings

__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"


class img2video:

    def __init__(imglist, fps, output='video.mp4', show=False, **kwargs):
        """
        img2video

        Make a video file from the images.
        Default output directory is the same as the images, but you can change the output directory from the 'output' parameter.

        Parameters
        ----------
        imglist : list
            List of image files.
        fps     : int
            Integer value of frame per second.
        output  : str
            Output video.
                * Default output directory is same as the file directory of image direcotry.
        show    : bool
            Show the animation.
                * Default is False.
        **kwargs : `~matplotlib.animation.FunctionAnimation.save`
            Keyword arguments of `~matplotlib.animation.FunctionAnimation.save` function (writer, dpi, codec, bitrate, extra_args, metadata, extra_anim, progress_callback).

        Returns
        -------
        video : str
            file name of the output video.

        Example
        -------
        >>> from fisspy import makevideo
        >>> from glob import glob
        >>> imglist = glob('/data/img/*.png')
        >>> makevideo.img2video(imglist, 10, 'video.mp4')
        """

        dirn = dirname(output)
        fname = basename(output)
        if not dirn:
            dirn = dirname(imglist[0])
            if not dirn:
                dirn = os.getcwd()
        fname = join(dirn, fname)

        # plot initial figure
        if not show:
            plt.ioff()
        img = plt.imread(imglist[0])
        ny, nx, nc =  im.shape
        self.fig, self.ax = plt.subplots(figsize=(nx/100, ny/100), dpi=100)
        self.im = self.ax.imshow(img)
        ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1, 0)
        ani = FunctionAnimation(fig, self.chImg, len(imglist),
                                interval=100, blit=False)
        ani.save(output, fps=fps, **kwargs)
        if not show:
            plt.close(fig)
            plt.ion()

        return fname

    def chImg(self, i):
        img = plt.imread(imglist[i])
        self.im.set_data(img)


def ffmpeg(imglist,fpsi,output='video.mp4'):
    """
    FFMPEG

    Using the ffmpeg make a video file from images.
    The output video is saved at the same location of images.

    Parameters
    ----------
    imglist : list
        List of image filename.
    fpsi   : int
        Integer value of Frame Per Second.
    output : str
        Output video name with extension.
            * Default is video.mp4

    Returns
    -------
    video : file
        Video file.

    Example
    -------
    >>> import fisspy
    >>> fisspy.ffmpeg(file,10,'fiss.mp4')
    """
    warnings.warn("As of v0.9.75, the `fisspy.makevideo.ffmpeg` function is depricated and will be removed in a v.1.0.0 version."
                  "Use the `img2video` or `func2video` functions to make video", Warning)
    FFMPEG_BIN = "ffmpeg"

    exten=output.split('.')[-1]
    if exten == 'mp4':
        codec='libx264'
    elif exten == 'avi':
        codec='libxvid'
    elif exten == 'mov':
        codec='mpeg4'
    else:
        ValueError('The given output extension is not supported !')

    n=len(imglist)
    if n == 0:
        raise ValueError('Image list has no element!')

    fps=str(fpsi)
    img=plt.imread(imglist[0])
    size=img.shape
    xsize=size[0]
    ysize=size[1]
    if bool(np.mod(xsize,2) + np.mod(ysize,2)):
        if np.mod(xsize,2):
            xsize -= 1
        if np.mod(ysize,2):
            ysize -=1
        for i in imglist:
            img = Image.open(i)
            img = img.resize([ysize,xsize])
            img.save(i)

#    if bool(np.mod(xsize,2)+np.mod(ysize,2)):
#        raise ValueError("The size of the image shuld be even numbers.")

    newname=np.arange(n)
    newname=np.char.add('_',newname.astype(str))
    newname=np.char.add(newname,'.png')

    dir=os.path.dirname(imglist[0])
    if bool(dir):
        os.chdir(dir)
    else:
        os.chdir(os.getcwd())

    f=open('img_list.tmp','w')
    for i in imglist:
        f.write("file '"+os.path.basename(i)+"'\n")
    f.close()

    cmd=(FFMPEG_BIN+
         ' -r '+fps+' -f concat -i img_list.tmp'+
         ' -c:v '+codec+' -pix_fmt yuv420p -q:v 1 -y '+output)

    res = os.system(cmd)
    os.remove('img_list.tmp')
    return res
