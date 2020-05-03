"""
Makevideo

Using the ffmpeg make a video file from images.

HISTORY
-------
* Nov 08 2016: Make a video by using ffmpeg program
* Jan 22 2020: Make a video by using matplotlib animation module
"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from os.path import dirname, basename, join


__author__ = "Juhyung Kang"
__email__ = "jhkang@astro.snu.ac.kr"


   
class img2video:

    def __init__(self, imglist, fps, output='video.mp4', show=False, **kwargs):
        """
        img2video

        Make a video file from the set of images.
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
        print("----Start to make video----")
        print(f"Frame per seconds: {fps}")
        
        self.imglist = imglist
        dirn = dirname(output)
        fname = basename(output)
        if not dirn:
            dirn = dirname(imglist[0])
            if not dirn:
                dirn = os.getcwd()
        fname = join(dirn, fname)
        print(f"Output: '{fname}'")
        
        # plot initial figure
        if not show:
            plt.ioff()
        img = plt.imread(imglist[0])
        ny, nx, nc =  img.shape
        self.fig, self.ax = plt.subplots(figsize=(nx/100, ny/100), dpi=100)
        self.im = self.ax.imshow(img, animated=True)
        self.ax.axis('off')
        self.fig.subplots_adjust(0, 0, 1, 1, 0)
        ani = FuncAnimation(self.fig, self.chImg, len(imglist),
                            interval=1e3/fps, blit=False)
        if show:
            plt.pause(0.1)
            
        ani.save(output, fps=fps, **kwargs)
        if not show:
            del ani
            plt.close(self.fig)
            plt.ion()
        print("----Done----")


    def chImg(self, i):
        img = plt.imread(self.imglist[i])
        self.im.set_data(img)


