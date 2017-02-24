============
Installation
============

FISSPy is a Python package for analysis NST/FISS data. 
FISSPy is python 2.7.x and 3.5.x compatible. 
The supporting python version depends on the required packages.

.. note:: It doesn't support the newest python 3.6.x,
          since the required pacakges `sunpy <http:://sunpy.org>`_ and 
          `interpolation <https://github.com/econforge/interpolation.py>`_ 
          don't support the python 3.6.x.\n
          
If you are new to Python or FISSPy then follow this guid to setup the scientific python and FISSPy. 
If you already familiar with the scientific Python, and hope to contribute or to improve to FISSPy 
package, see the :ref:`development` section.

.. _main-install:

Installing Python package manager
---------------------------------

The `Anaconda package manager <https://anaconda.org/>`_ (called also Anaconda Python distribution) 
is very powerful to install and manage scientific Python packages. 
If you already installed the Anaconda package manager, and want to just install the FISSPy skip to
the :ref:`fisspy-install`.

Anaconda contains a Python and a lot of scientific packages. Anaconda provides (almost) all the
packages which is needed to use FISSPy, but FISSPy is not provided by Anaconda yet. FISSPy will
be uploaded on the Anaconda package server soon.

To install the Anaconda package manager, visit 
`Anaconda install website <https://www.continuum.io/downloads>`_. You just download the installer
which is fit to your operating system and follow the instruction of installer. Note in the anaconda
website there are two option the Python, 2.7 or 3.x versions. Since some of packages do not support
Python 3.6 or Python 3.5, we recommend that users install the Python **2.7** or **3.5**, not 3.6. However 
if all of the required packages will be supported on the latest Python 3.x version of Anaconda, we
also support FISSPy on the latest Python 3.x.

.. warning::
    FISSPy do not support on Python 3.6 version yet.\n
    Please download the Python 2.7 or 3.5 versions.

.. note::
    If `sunpy <http:://sunpy.org>`_ and `interpolation <https://github.com/econforge/interpolation.py>`_
    can be supported on Python 3.6 version, above warning and this note boxes will be removed.

.. note::
    The Python 2.7 is scheduled to be deprecated in 2020.

.. _fisspy-install:

Installing FISSPy
#################

If you did not download Anaconda package manager, we recommend that see the :ref:`Anaconda installation 
<mail-install>` section first.

To install FISSPy::

    pip install fisspy

This `pip` command accesses to `PyPI (Python Package Index) <https://pypi.python.org/pypi>`_
website, and installs the FISSPy on your computuer automatically, also the required packages.

**Done! Congraturation!!**

Upgrading FISSPy to a New Version
#################################

If there is a new version of released FISSPy you can upgrade to the latest version like this::

    pip install fisspy -U


Development version Installation
--------------------------------

If you wish to contribute to the fisspy development and debug, refer the following development
version installation.

.. toctree::
  :maxdepth: 2

  develop.rst
