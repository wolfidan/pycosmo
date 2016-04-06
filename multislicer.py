# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:54:37 2015

@author: wolfensb
"""
from slicer import slicer
import numpy as np

def multislicer(list_vars, type, idx, method='nearest'):
    slices=[]
    if type=='PPI':
        for i, var in enumerate(list_vars):
            print i
            if i==0:
                slices.append(slicer(var,type,idx,method))
                fast_reslice=slices[0].fast_reslice
            else:
                size_model=var.data[0,:,:].shape
                size_PPI=slices[0].data.shape
                indexes_slice=fast_reslice['idx_slice']
                indexes_model=fast_reslice['idx_model']
                list_closest=fast_reslice['list_closest']
                delta_dist_list=fast_reslice['delta_dist_list']
                data_interp=np.zeros((len(indexes_model),))*float('nan')
                for j, idx_slice in enumerate(indexes_slice):
                    closest=list_closest[j]
                    delta_dist=delta_dist_list[j]
                    idx_model=indexes_model[idx_slice]
                    idx_model_2D=np.unravel_index(idx_model,size_model)
                    data_model=var.data[:,idx_model_2D[0],idx_model_2D[1]]
                    try:
                        if len(closest)==1:
                            data_interp[i]=data_model[closest]
                        else:
                            data_interp[idx_slice]=data_model[closest[1]]+(data_model[closest[0]]-data_model[closest[1]])*delta_dist
                    except:
                        raise
                
                data_PPI=np.reshape(data_interp, size_PPI)
                slice=var.copy() # copy original variable to slice
                slice.dim = slice.dim
                slice.data = data_PPI
                slice.coordinates.pop('lon_2D',None)
                slice.coordinates.pop('lat_2D',None)
                slice.coordinates['lon_2D_PPI']=slices[0].coordinates['lon_2D_PPI']
                slice.coordinates['lat_2D_PPI']=slices[0].coordinates['lat_2D_PPI']
                slice.attributes['altitudes']=slices[0].attributes['altitudes']
                slice.name+='_PPI_SLICE'
                slice.coordinates.pop('hyb_levels',None) # This key is not needed anymore
                slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
                slice.attributes.pop('z-levels',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
                
                slices.append(slice)
                
    return slices
    
if __name__=='__main__':
    import pycosmo as pc
    import matplotlib.pyplot as plt
    import time
    # Open a file, you have to specify either grib (.grb, .grib) or netcdf (.nc, .cdf) files as input, if no suffix --> program will assume it is grib
    file_h=pc.open_file('/ltedata/COSMO/case2014040814_PAYERNE_analysis_full_domain/lfsf00155000')
#    print file_nc # Shows all variables contained in file
#    print file_nc.attributes # Global attributes
#    
    # Get cloud top
    T=pc.get_variable(file_h,'T')
    T.assign_heights()
    index_PPI={'range':100000, 'radar_pos':[46.36,7.22,3000],'resolution':500, 'elevation': 1}
    
    start_time = time.time()
    T_slice=multislicer([T,T,T,T,T],'PPI',index_PPI)