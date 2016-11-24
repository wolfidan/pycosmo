#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# interp1 extension module
_interp1_c = Extension("_interp1_c",
                   ["./pycosmo/c/interp1_c.i","./pycosmo/c/interp1_c.c"],
                   include_dirs = [numpy_include],
                   )
_radar_interp_c = Extension("_radar_interp_c",
                   ["./pycosmo/c/radar_interp_c.i","./pycosmo/c/radar_interp_c.c"],
                   include_dirs = [numpy_include],
                   )
# ezrange setup
setup(  name        = "pycosmo",
        description = "Python tools for the COSMO NWP model",
        version     = "1.0",
        url='http://gitlab.epfl.ch/wolfensb/radar_simulator/',
        author='Daniel Wolfensberger - LTE EPFL',
        author_email='daniel.wolfensberger@epfl.ch',
        license='GNU',
        packages=['pycosmo'],
        include_package_data=True,
        install_requires=[
          'pyproj',
          'numpy',
          'scipy',
	  'basemap',
        ],
        zip_safe=False,
        ext_modules = [_interp1_c,_radar_interp_c ]
        )


