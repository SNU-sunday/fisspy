#!/usr/bin/env python

import glob
import os
import sys

from setuptools import setup, find_packages


# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', 'FISSPy: Python for GST/FISS instruement')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'BSD-2')
URL = metadata.get('url', 'http://fiss.snu.ac.kr')

LONG_DESCRIPTION = "FISSPy is the python packages to analyze the GST/FISS data file."

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.8.0'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION


extras_require = {'database': ["sqlalchemy"],
                  'image': ["scikit-image"],
                  'jpeg2000': ["glymur"],
                  'net': ["suds-jurko", "beautifulsoup4", "requests"]}
extras_require['all'] = extras_require['database'] + extras_require['image'] + \
                        extras_require['net'] + ["wcsaxes>=0.8"]

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=find_packages(exclude=['docs']),
      install_requires=['numpy>1.7.1',
                        'astropy>=1.3',
                        'pandas>=0.12.0',
                        'matplotlib>=2.0',
                        'sunpy>=0.7.6',
			'scipy',
			'interpolation',
                        'statsmodels>=0.6.0',
			'suds-jurko'],
      extras_require=extras_require,
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      use_2to3=False,
      include_package_data=True,
      )
