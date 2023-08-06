"""FISSPy sample data file set"""
from __future__ import absolute_import, print_function

from astropy.utils.data import download_file
from sunpy.util.net import url_exists
import os.path
from os import mkdir
from shutil import move

__author__ = "Juhyeong Kang"
__email = "jhkang@astro.snu.ac.kr"


url='http://fiss.snu.ac.kr/sample-data/'
files=['FISS_20140603_164842_B1_p.fts',
      'mFISS_20140603_164842_B1_c.fts']

home=os.path.expanduser("~")
if not home:
    raise RuntimeError('Environment variable $HOME is not defined.')

fissdir=os.path.join(home,'fisspy')
if not os.path.isdir(fissdir):
    mkdir(fissdir)

sampledir=os.path.join(fissdir,'sample_data')
if not os.path.isdir(sampledir):
    mkdir(sampledir)

def download_sample_data():
    """
    Download the sample data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print("Downloading sample fiss files to {}".format(sampledir))
    for f in files:
        if url_exists(url+f):
            df = download_file(url+f)
            move(df,os.path.join(sampledir,f))
