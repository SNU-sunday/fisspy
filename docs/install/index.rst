============
Installation
============

FISSPy is a Python package for analysis GST/FISS data. 
FISSPy is python 2.7.x and 3.5.x compatible. 
The supporting python version depends on the required packages.

.. note:: The FISSPy package highly depends on the `sunpy <http:://sunpy.org>`_
          and the `interpolation <https://github.com/econforge/interpolation.py>`_ packages.
          
          
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
website there are two option the Python, 2.7 or 3.x versions. Since the Python **2.7** is supported 
until 2020, we recommend that users install the latest Python **3.x** rather than **2.7**.

.. note::
    The Python 2.7 is scheduled to be deprecated in 2020.

.. _fisspy-install:

Installing FISSPy
#################

If you did not download Anaconda package manager, we recommend that see the :ref:`Anaconda installation 
<mail-install>` section first.

To install FISSPy, you first configure the conda cahnnel::

    conda config --add channels conda-forge

then install FISSPy::    
    
    conda install -c kailia0209 fisspy
    
Also you can intsall FISSPy by using pip command(not recommended)::
    
    pip install fisspy

These two way install the FISSPy on your computuer automatically, also the required packages.

.. note::
    If the requried packages are not installed, then you should install the requried packages manually.

**Done! Congraturation!!**

Required Packages list
######################

* `astropy <http://www.astropy.org/>`_
* `interpolation <https://github.com/econforge/interpolation.py>`_
* `numpy <http://www.numpy.org/>`_
* `pandas <http://pandas.pydata.org/>`_
* `scipy <https://www.scipy.org/>`_
* `statsmodels <http://www.statsmodels.org/stable/index.html>`_
* suds-jurko
* `sunpy <http:://sunpy.org>`_

Upgrading FISSPy to a New Version
#################################

If there is a new version of released FISSPy you can upgrade to the latest version like this::

    conda update fisspy
    
or::    

    pip install fisspy -U


Development version Installation
--------------------------------

If you wish to contribute to the fisspy development and debug, refer the following development
version installation.

.. toctree::
  :maxdepth: 2

  develop.rst
