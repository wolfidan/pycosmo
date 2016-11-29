# pycosmo
A python 2.7 library for reading/manipulating/plotting GRIB or NetCDF files from the COSMO NWP model. Pycosmo was developped
on linux and tested with linux only. As such it is expected NOT to work on windows without significant modifications

# Installation

## External dependencies
You need to have at least a working installation of the swig, netCDF, geos and grib2 libraries 

You can install them by running

sudo apt-get install libnetcdf-dev libnetcdf libgrib-api-dev swig libgeos-dev

## Python dependencies
The pycosmo library relies on numpy, scipy, matplotlib, pyproj, pynio and (optional) basemap

You can install  numpy, scipy, matplotlib and pyproj by simply running

sudo apt-get install python-pip
sudo pip install numpy scipy matplotlib pyproj

### Pynio
For pynio (IO of GRIB files) you need to download the source from

https://www.earthsystemgrid.org/dataset/pynio.1.4.1.0.html

Then unzip the archive and compile the library with

export F2CLIBS=name_of_your_fortran_compiler; export F2CLIBS_PREFIX=path_to_your_libfortran_so_files; CFLAGS='-Wno-error=format-security'; export NCL_GRIB_PTABLE_PATH=path_to_your_grib_keys_folder; sudo -E python setup.py install

Indeed pynio needs some environment variables to be defined to work properly. NCL_GRIB_PTABLE_PATH indicates the path of the grib keys required by Pynio to properly read COSMO files.
These keys can be downloaded from http://www.ncl.ucar.edu/Applications/Files/gt.tar and are also given in your pycosmo folder (folder cosmo_grib_keys). As an example, on my computer the installation command is

export F2CLIBS=gfortran; export F2CLIBS_PREFIX=/usr/local/lib/; CFLAGS='-Wno-error=format-security'; export NCL_GRIB_PTABLE_PATH=~/pycosmo/pycosmo/cosmo_grib_keys/; sudo -E python setup.py install

### Basemap (optional)

Basemap allows to plot georeferenced data on maps. It is optional, if you want to install it you need to download it from

https://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/

Then unzip the archive and start by compiling its dependency geos, go to the geos-3.3.3 folder inside the basemap main folder and run

./configure --prefix==/usr/local/
make
sudo make install

Then return to the main basemap folder and install the library with

sudo python setup.py install

## Compilation

Download tar from github, unzip and run the setup.py file

git clone https://github.com/wolfidan/pycosmo.git
cd pycosmo
sudo python setup.py install

# Usage
Check the examples folder for examples of use
