# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:56:58 2015

@author: wolfensb
"""

import PyNIO.Nio as Nio 
import sys, os


class File_class:
    def __init__(self,fname):
        bname=os.path.basename(fname)
        name, extensionname = os.path.splitext(bname)
        last_letter=name[-1]
        # File-type can be either h (hybrid), p (pressure) and z (height), in COSMO it is indicated with the last letter of the filename
        if last_letter in ['p','c','z']: file_type=last_letter
        else: file_type='h'
        print '--------------------------'
        print 'Reading file '+fname
        if not os.path.exists(fname):
            print 'File not found! Aborting...'
            sys.exit(0)
        if extensionname=='':
            print 'Input file has no extension, assuming it is GRIB...'
            file_format='grib'
            fname=fname+'.grb'
        elif extensionname in ['.gr', '.gr1', '.grb', '.grib', '.grb1', '.grib1', '.gr2', '.grb2', '.grib2']:
            print 'Input file is a grib file'
            file_format='grib'
        elif extensionname in ['.nc', '.cdf', '.netcdf', '.nc3', '.nc4']:
            print 'Input file is NetCDF file'
            file_format='ncdf'
        else:
            print 'Invalid data type, must be GRIB or NetCDF, aborting...'
            sys.exit(0)          

        fhandle = Nio.open_file(fname, 'r')
        
        glob_attr={}
        for name in fhandle.attributes.keys():
            glob_attr[name]=getattr(fhandle,name)
    
        self.handle=fhandle                                                                                                     
        self.name=fname
        self.format=file_format
        self.type=file_type
        self.attributes=glob_attr
        self.dic_variables={}
        
        print 'File '+fname+' read successfully'
        print '--------------------------'
        print ''
        
    def close(self):
        self.handle.close()
        self.dic_variables={}
        
    def __str__(self):
        output = self.handle.__str__()
        return output
