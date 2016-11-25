# pycosmo
A python 2.7 library for reading/manipulating/plotting GRIB or NetCDF files from the COSMO NWP model. Pycosmo was developped
on linux and tested with linux only. As such it is expected NOT to work on windows without significant modifications

# Installation

## External dependencies
You need to have at least a working installation of the swig library, the netCDF library 
as well as a working installation of the GRIB2 library 

You can install them by running

sudo apt-get install libnetcdf-dev libnetcdf libgrib-api-dev swig

## Python dependencies
The pycosmo library relies on numpy, scipy, matplotlib and pyproj

You can install  numpy, scipy, matplotlib and pyproj by running

sudo apt-get install python-pip
sudo pip install numpy scipy matplotlib basemap pynio pyproj

For basemap and pynio you need miniconda so run

wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh

Enter yes to everything that is asked, then run

~/miniconda2/bin/conda install --channel ncar pynio
~/miniconda2/bin/conda install basemap

## Compilation

Download tar from github, unzip and run the setup.py file

git clone https://github.com/wolfidan/pycosmo.git
cd pycosmo
sudo python setup.py install

# Usage
Check the examples folder for examples of use
