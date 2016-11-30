# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:56:58 2015

@author: wolfensb
"""

from pycosmo.derived_vars import DERIVED_VARS, get_derived_var
import pycosmo.data as d

import PyNIO.Nio as Nio 
import numpy as np
import os
import warnings
from fnmatch import fnmatch

_nio_builtins = ['__class__', '__delattr__', '__doc__', '__getattribute__', '__hash__', \
            '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', \
            '__setattr__', '__str__', '__weakref__', '__getitem__', '__setitem__', '__len__' ]

_nio_localatts = ['attributes','_obj','variables','file','varname', \
              'cf_dimensions', 'cf2dims', 'ma_mode', 'explicit_fill_values', \
              'mask_below_value', 'mask_above_value', 'set_option', 'create_variable', 'close', 'assign_value', 'get_value'] 
    
def open_file(fname): # Just create a file_instance class
    return FileClass(fname)
    
def get_grib_keys():
    # Reads the grib keys names from the grib_keys.txt file
    cur_path=os.path.dirname(os.path.realpath(__file__))
    f = open(cur_path+'/grib_keys.txt', 'r')
    dic={}
    for line in f:
        line=line.strip('\n')
        line=line.split(',')
        dic[line[0]]=line[1]
    return dic
    
class FileClass(object):
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
            warnings.warn('File not found! Aborting...')
            return []
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
            warnings.warn('Invalid data type, must be GRIB or NetCDF, aborting...' )     

        _fhandle = Nio.open_file(fname, 'r')
    
        self._handle = _fhandle                                                                                                     
        self.name=fname
        self.format=file_format
        self.type=file_type
        self.dic_variables={}

        print 'File '+fname+' read successfully'
        print '--------------------------'
        print ''

        
    def __getattribute__(self, attrib):
        if attrib in _nio_builtins or attrib in _nio_localatts:
            return self._handle.__getattribute__(attrib)
        else:
            return object.__getattribute__(self,attrib)
            
    def close(self):
        self._handle.close()
        self.dic_variables={}
        
    def __str__(self):
        output = self.__getattribute__('__str__')()
        return output

    def check_varname(self,varname):
        varname_checked=''    
        # First try to find variable in file_instance
        list_vars = np.asarray((self.variables.keys()))
    
        if varname in list_vars:
            varname_checked = varname
        else:  # Then try to find it using the grib key dictionary
            dic = get_grib_keys()
            if varname in dic.keys():
                grib_varname=dic[varname]
                match=list_vars[np.where([fnmatch(l,grib_varname) for l in list_vars])[0]]
                if len(match) > 1:
                    # Several matches were found in the file_instance, we will keep only the ones that do not match with any other key in the grib key dictionary
                    all_grb_keys=dic.values()
                    match_check=[]
                    for m in match:
                        if not any([fnmatch(l,m) for l in all_grb_keys]): # Check if other keys 
                            match_check.append(m)
                    if(match_check):
                        varname_checked=match_check[0]
                    else:
                        varname_checked=match[0]
                elif len(match) == 1: # If only one match, use that one
                    varname_checked=match[0]
            else: # Try to see if varname was entered with a wildcard
                match=list_vars[np.where([fnmatch(l,varname) for l in list_vars])[0]]
                if len(match) >= 1:
                    varname_checked=match[0]
                
        return varname_checked
    
    def get_variable(self, var_names, get_proj_info=True, assign_heights=False,
                     shared_heights=False, cfile_name=''):
        
        # Create dictionary of options
        import_opts = {'get_proj_info':get_proj_info,\
                       'shared_heights':shared_heights,\
                       'assign_heights':assign_heights,\
                       'cfile_name':cfile_name}

        if isinstance(var_names,list):
            dic_var={}
            for i,v in enumerate(var_names):
                var = self.get_variable(v, **import_opts)
                if assign_heights:
                    if i > 0 and shared_heights:
                        # Stop assigning heights, after first variable
                        import_opts['assign_heights'] = False
                        # If shared_heights is true we just copy the heights from the first variables to all others
                        var.attributes['z-levels'] = dic_var[var_names[0]].attributes['z-levels']
                dic_var[v]=var
            return dic_var
        else:
            print('--------------------------')
            print('Reading variable '+var_names)
            if var_names in self.dic_variables.keys():
                var = self.dic_variables[var_names]
            elif var_names in DERIVED_VARS:
                var = get_derived_var(self,var_names,import_opts)
            else:
                varname_checked = self.check_varname(var_names)
                if varname_checked != '':
                    var = d.DataClass(self, varname_checked, get_proj_info)
                else:
                    print('Variable was not found in file_instance')
                    return

            self.dic_variables[var_names] = var
                    
            # Assign heights if wanted
            if assign_heights and var:
                var.assign_heights(cfile_name)
            print 'Variable was read successfully'
            print('--------------------------' )
            print('')
            return var 

    def check_if_variables_in_file(self, varnames):
        for var in varnames:
            varname_checked = self.check_varname(var)
            if varname_checked == '':
                return False
        return True
       
            
if __name__ == '__main__':
    import pycosmo as pc
    file_h = pc.open_file('/ltedata/COSMO/Validation_operator/case2014110500_ONEMOM/lfsf00135500')
    
    U = file_h.get_variable('U')
    U.assign_heights(cfile_name='/ltedata/COSMO/Validation_operator/case2014110500_ONEMOM/lfsf00000000c')
    T = file_h.get_variable('T')
    T.assign_heights(cfile_name='/ltedata/COSMO/Validation_operator/case2014110500_ONEMOM/lfsf00000000c')    
#    
#    P = file_h.get_variable('P')
#    
#    li = file_h.get_variable(['P','T'],shared_heights=False,assign_heights=True)