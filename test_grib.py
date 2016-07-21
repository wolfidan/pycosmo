# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:09:42 2016

@author: daniel
"""

import numpy as np
import glob
import os
import warnings

import pygrib
import PyNIO.Nio as Nio 

# Read the grib tables


def read_grb_tables(folder = './gt/'):
    tables = {}
    tab_filenames = glob.glob(folder+'/*.gtb')
    for tab in tab_filenames:
        
        tab_code = os.path.basename(tab).strip('.gtb')
        tables[tab_code] = {}
        
        with open(tab,'r') as f:
            all_lines = f.readlines()
            for line in all_lines[5:]:
                line_split = line.split(':')
                key = int(line_split[0])
                values = [l.strip('\t').strip('\n') for l in line_split[1:]]
                tables[tab_code][key] = values
            
    return tables
    
def grb_to_grid(grb_obj):
    """Takes a single grb object containing multiple
    levels. Assumes same time, pressure levels. Compiles to a cube"""
    n_levels = len(grb_obj)
    levels = np.array([grb_element['level'] for grb_element in grb_obj])
    indexes = np.argsort(levels)[::-1] # highest pressure first
    cube = np.zeros([n_levels, grb_obj[0].values.shape[0], grb_obj[0].values.shape[1]])
    for i in range(n_levels):
        cube[i,:,:] = grb_obj[indexes[i]].values
    
    parameter_id = grb_obj[0].indicatorOfParameter
    parameter_table_version = 'mch_'+str(grb_obj[0].table2Version).zfill(3)
    
    try:
        name = tab[parameter_table_version][parameter_id][0]
        units = tab[parameter_table_version][parameter_id][1]
        long_name = tab[parameter_table_version][parameter_id][2]
    except:
        name = grb_obj[0].shortName
        units = grb_obj[0].units
        long_name = grb_obj[0].name  
        warnings.warn('Unknown grib parameter number detected (%d), \
                      table version (%s), using default grib name (%s)'%(parameter_id,
                      parameter_table_version,name))
                       
    cube_dict = {'data' : np.squeeze(cube), 'units' : units, 'name':name,
                 'long_name':long_name, 'levels' : levels[indexes], 
                 'parameter_table_version' : parameter_table_version}
    return cube_dict
        
    
tab = read_grb_tables()

#
grbs = pygrib.open('/ltedata/COSMO/Validation_operator/case2014110500_ONEMOM/lfsf00120000')
#
current_par = grbs[1].indicatorOfParameter
grb_obj = [grbs[1]]
dic = {}
for i,grb in enumerate(grbs[1:]):

    if grb.indicatorOfParameter == current_par:
        grb_obj.append(grb)
    else:
        
        grid = grb_to_grid(grb_obj)

        dic[grid['name']] = grid

        current_par = grb.indicatorOfParameter
        grb_obj = [grb]
#        
#    

grbNio = Nio.open_file('/ltedata/COSMO/Validation_operator/case2014110500_ONEMOM/lfsf00120000.grb','r')