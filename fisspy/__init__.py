"""
FISSPy
======

An free and open-source Python package for `GST/FISS <http://fiss.snu.ac.kr>`_ instrument.

Links
-----
Homepage : http://fiss.snu.ac.kr \n
Documentation : http://fiss.snu.ac.kr/fisspy
"""

from __future__ import absolute_import
__author__="SNU Solar Group"
__version__="1.1.1"


from . import cm, makevideo, analysis, align, read, correction, preprocess, image, data
from .image.raster_set import makeRasterSet