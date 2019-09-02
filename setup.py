from setuptools import setup, find_packages

setup(
    name='fisspy',
    version='0.9.70',
    description='fisspy: Python analysis tools for GST/FISS',
    url='http://fiss.snu.ac.kr',
    author='Juhyung Kang',
    author_email='jhkang@astro.snu.ac.kr',
    license='BSD-2',
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs', 'logo']),
    install_requires=["numba", "numpy", "scipy", "astropy>=3.0",
                      "sunpy>=1.0.0", "pandas", "matplotlib>=3.0",
		      "interpolation>=2.0", "statsmodels",
		      "suds-jurko", "pillow", "ffmpeg"],
    zip_safe=False
    )

