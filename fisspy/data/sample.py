"""FISSPy sample data files"""
from __future__ import absolute_import

from ._sample import sampledir, files, download_sample_data
import sys
from os.path import isfile, join

if isfile(join(sampledir, files[1])):
    setattr(sys.modules[__name__],'FISS_IMAGE',join(sampledir,files[1]))
else:
    download_sample_data() 
    setattr(sys.modules[__name__],'FISS_IMAGE',join(sampledir,files[1]))
