from setuptools import setup, find_packages

setup(
    name='fisspy',
    version='1.1.1',
    description='fisspy: Python analysis tools for GST/FISS',
    url='http://fiss.snu.ac.kr',
    author='Juhyung Kang',
    author_email='jhkang@astro.snu.ac.kr',
    license='BSD-2',
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs', 'logo']),
    install_requires=["numba", "numpy", "scipy>=1.5", "astropy>=5.0", "sunpy>=2.0.0", "matplotlib>=3.0", "interpolation>=2.2.6", "statsmodels", "beautifulsoup4", "pandas", "ffmpeg", "pyqt>=5.0"],
    zip_safe=False
    )
