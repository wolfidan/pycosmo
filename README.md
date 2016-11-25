# pycosmo
A python 2.7 library for reading/manipulating/plotting GRIB or NetCDF files from the COSMO NWP model. Pycosmo was developped
on linux and tested with linux only. As such it is expected NOT to work on windows without significant modifications

# Installation

## External dependencies
You need to have at least a working installation of the netCDF library 
as well as a working installation of the GRIB2 library 

You can install them by running

sudo apt-get install libnetcdf-dev libnetcd
sudo apt-get install libgrib-api-dev

## Python dependencies
The pycosmo library relies on numpy, scipy, matplotlib, basemap, pynio and pyproj

You can install all dependencies by running

sudo apt-get install python-pip
sudo pip install numpy scipy matplotlib basemap pynio pyproj

## Compilation

Download tar from github, unzip and run the setup.py file

git clone https://github.com/wolfidan/pycosmo.git
cd pycosmo
sudo python setup.py install

# Usage
Check the examples folder for examples of use
