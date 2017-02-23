"""FISSPy sample data files"""
from __future__ import absolute_import

from ._sample import sampledir, files
import sys
from os.path import isfile, join

if isfile(join(sampledir, files[1])):
    setattr(sys.modules[__name__],'FISS_IMAGE',join(sampledir,files[1]))
else:
    raise ImportError("Sample data missing. Please download the sample file by using fisspy.data.download_sample_data()")
    