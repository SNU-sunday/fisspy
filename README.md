# FISSpy

[![Latest Version](https://img.shields.io/pypi/v/fisspy.svg)](https://pypi.python.org/pypi/fisspy/) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/fisspy/badges/version.svg)](https://anaconda.org/conda-forge/fisspy) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/fisspy/badges/downloads.svg)](https://anaconda.org/conda-forge/fisspy)

The Python Package for data analysis of the [GST/FISS instrument](http://fiss.snu.ac.kr/).

Installation
------------

Requirement Package:

* [Python](http://www.python.org) >=3.6
* [NumPy](http://numpy.scipy.org/)
* [Matplotlib](http://matplotlib.sourceforge.net/) >=3.0
* [SciPy](http://www.scipy.org/) 
* [sunpy](http://sunpy.org/) >=2.0.0
* [Astropy](http://astropy.org)
* [Interpolation](https://github.com/EconForge/interpolation.py) >=2.0.0

Recommend to install the Python from the [Anaconda](https://www.continuum.io/why-anaconda).

To install these packages, first set the conda-forge server:

    conda config --append channels conda-forge
    
### <span style="color: red">Option 1)</span> Using Mamba
We highly recommend you use the mamba to install the FISSpy:

    mamba install fisspy

If you don't have mamba please see [mamba installation](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

After installing the mamba you can download the FISSpy using the previous command.

### <span style="color: red">Option 2)</span> Using Conda
Sometimes an old version of the conda has some errors and takes lots of time to download the FISSpy. If then we recommend you use the mamba instead See [Option 1](#option-1-using-mamba).

    conda install fisspy

### <span style="color: red">Option 3)</span> Using PyPI
If you failed to download the FISSpy using the above options, you can download the FISSpy using PyPI, but it is not the recommended method.

    pip install fisspy


Tutorials
---------
You can see the tutorials for the FISSpy on [here](http://fiss.snu.ac.kr/fisspy/).
