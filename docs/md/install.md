---
id: "my-style"
---

@import "fisspy_guide.less"

# Installation

[![Latest Version](https://img.shields.io/pypi/v/fisspy.svg)](https://pypi.python.org/pypi/fisspy/) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/fisspy/badges/version.svg)](https://anaconda.org/conda-forge/fisspy) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/fisspy/badges/downloads.svg)](https://anaconda.org/conda-forge/fisspy)

If you are new to Python please see the [Python Tutorial](/fisspy/python), and install [Anaconda](https://www.anaconda.com/).

## Requirements

* [Python](http://www.python.org) >= 3.6
* [Astropy](http://astropy.org) >= 2.0
* [Interpolation](https://github.com/EconForge/interpolation.py) >= 2.0.0

These required packages are automatically installed if you install fisspy by using [Anaconda](https://www.anaconda.com/).

## Installing fisspy using Anaconda

To install fisspy launch a terminal (under linux or OSX) or the 'Anaconda command Prompt' (under Windows). First add the conda-forge channels:

```
conda config --append channels conda-forge
```

Then install the fisspy:

```
conda install fisspy
```

You can also install the specific version of fisspy:

```
conda install fisspy=0.9.80
```

### Updating fisspy

You can update to the latest version by running:

```
conda update fisspy
```

## Installing fisspy using PyPI
You can also install fisspy by using PyPI, but we highly recommand to install fisspy by using Anaconda.

```
pip install fisspy
```
