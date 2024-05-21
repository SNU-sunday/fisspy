"""
FISSpy read fts file package
"""
from __future__ import absolute_import

from fisspy.io.read import *
import warnings

warnings.warn("As of v0.9.0, the `fisspy.io` module is deprecated "
              "and will be removed in a v1.0.0 version. "
              "Use `fisspy.read` to read the FISS data file.", Warning)