"""
FISSPy
======

An free and open-source Python package for `GST/FISS <http://fiss.snu.ac.kr>`_ instrument.

Links
-----
Homepage : http://fiss.snu.ac.kr \n
Documentation : http://docs.fisspy.
"""

from __future__ import absolute_import
__author__="SNU Solar Group"
__version__="0.8.0"


import fisspy.cm
from fisspy.makevideo import ffmpeg
from .image.interactive import IFDV
