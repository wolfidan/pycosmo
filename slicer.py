# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:58:00 2015

@author: wolfensb
"""
import pyproj as pyproj
import numpy as np
import scipy.spatial as spatial
from utilities import binary_search

# Two extreme earth radius
a=6378.1370*1000
b=6356.7523*1000
        
        
def slicer(var, type, idx, method='nearest'):
    # Type can be either:
        # "i" (scalar) : Cut at fixed zero-based i-index (no interpolation).
        # "j" (scalar) : Cut at fixed zero-based j-index (no interpolation).
        # "k" (scalar) : Cut at fixed zero-based k-index (no interpolation).
        # "level" (scalar) : Cut at fixed level (e.g. pressure in a p-file or height in a z-file or h-file) (interpolation)
        # "lat" (scalar) : Cut at fixed geographical latitude (interpolation)
        # "lon" (scalar) : Cut at fixed geographical longitude (interpolation)
        # "latlon" (nx2 array) : Cut along list of geographical latitude/longitude pairs (interpolation)
    
    
    # Check dimension of variable
    dim = var.dim
    # Check if variable has z-levels and if we should slice them:
    slice_z=False
    if 'z-levels' in var.attributes.keys() and dim == 3:
        slice_z=True
        
    if dim == 2 and (type == 'level' or type == 'k'):
        print 'Variable must be 3D to cut at vertical coordinates!'
        return
    if (var.file.type=='h' and 'z-levels' not in var.attributes.keys()) and type == 'level':
        print 'In order to cut at a fixed height for hybrid layer files, you need to first assign altitude levels using the assign_heights function...'
        return
        

    slice=var.copy() # copy original variable to slice
    
    if type == 'i': # Cut at fixed zero-based i-index (no interpolation)
        if dim == 2: slice.data=var.data[idx,:]
        elif dim == 3: slice.data=var.data[:,idx,:]
        if slice_z:
            slice.attributes.pop('z-levels',None)
            slice.attributes['z-levels']=var.attributes['z-levels'][:,idx,:]
        slice.coordinates['lat_2D']=var.coordinates['lat_2D'][idx]
        slice.dim = slice.dim - 1
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
        slice.name+='_i_SLICE'

    elif type == 'j': #  Cut at fixed zero-based j-index (no interpolation)
        if dim == 2: slice.data=var.data[:,idx]
        elif dim == 3: slice.data=var.data[:,:,idx]
        if slice_z:
            slice.attributes.pop('z-levels',None)
            slice.attributes['z-levels']=var.attributes['z-levels'][:,:,idx]
        slice.coordinates['lon_2D']=var.coordinates['lon_2D'][idx]
        slice.dim = slice.dim - 1
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field         
        slice.name+='_j_SLICE'

    elif type == 'k': #  Cut at fixed zero-based k-index (no interpolation)
        slice.data=var.data[:,:,idx]
        try:
            slice.coordinates['heights']=var.coordinates['heights'][idx,:,:]
        except: # variable has not been georeferenced vertically...
             slice.coordinates['hyb_levels']=var.coordinates['hyb_levels'][idx]
        if slice_z:
            slice.attributes['z-levels']=var.attributes['z-levels'][idx,:,:]
            
        slice.name+='_k_SLICE'
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
        slice.dim = slice.dim - 1
        
    elif type == 'level':
        cut_val=idx
        siz=var.data.shape
        if var.file.type == 'p' or var.file.type == 'z':
            if var.file.type == 'p':
                print 'File is a p-file, trying to cut at p='+str(idx)+'hPa...'
                coord_name='press_levels'
            elif var.file.type == 'z' :
                print 'File is a z-file, trying to cut at z='+str(idx)+'m...'
                coord_name='height_levels'
    
            coords=var.coordinates[coord_name]
            
            if not cut_val in coords: # Linear interpolation
                closest=pc.binary_search(coords,cut_val)
                if len(closest)==1:
                    data_interp=var.data[closest,:,:]
                else:
                    dist=coords[closest[0]]-coords[closest[1]]
                    data_interp=var.data[closest[1],:,:]+(var.data[closest[0],:,:]-var.data[closest[1],:,:])/dist*(cut_val-coords[closest[1]])
            
            else:
                idx_cut=np.where(coords == cut_val)[0]
                data_interp=var.data[idx_cut[0],:,:]
            
        elif var.file.type == 'h':
            coord_name='hyb_levels'
        
            zlevels=var.attributes['z-levels']
            arr=(zlevels-cut_val)
            arr[arr<0]=99999 # Do this in order to find only the indexes of heights > cut_val
            
            indexes_larger=np.argmin(arr,axis=0)
            indexes_smaller=indexes_larger+1 # remember that array is in decreasing order...
            indexes_smaller[indexes_smaller>siz[0]-1]=siz[0]-1
            
            k,j = np.meshgrid(np.arange(siz[2]),np.arange(siz[1])) # One of numpy's limitation is that we have to index with these index arrays (see next line)
            dist=zlevels[indexes_larger,j,k]-zlevels[indexes_smaller,j,k]
            data_interp=var.data[indexes_smaller,j,k]+(var.data[indexes_larger,j,k]-var.data[indexes_smaller,j,k])/dist*(cut_val-zlevels[indexes_smaller,j,k])
       
            # Now mask
            data_interp[indexes_smaller==siz[0]-1]=float('nan')
         
        slice.data=data_interp
        slice.attributes.pop(coord_name,None)
        slice.dim = slice.dim - 1
        slice.name+='_level_SLICE'
        # TODO !!!

    elif type == 'lat':
        if idx<var.attributes['domain_2D'][0][0] or idx>var.attributes['domain_2D'][1][0]:
            print 'Specified slicing latitude is outside of model domain!'
            return
            
        try:
            lat_2D=var.coordinates['lat_2D']
        except:
            lat_2D=var.coordinates['lat_2D_resized']
        
        try:
            lon_2D=var.coordinates['lon_2D']
        except:
            lon_2D=var.coordinates['lon_2D_resized']
            
        siz=var.data.shape
        if dim == 3:
            siz_hor=siz[1:3] # We're not interested in the vertical levels yet
            nVertLayers=siz[0]
        else: # variable is 2D
            siz_hor=siz
            nVertLayers=1

        slice_data=np.zeros((nVertLayers,siz_hor[1]))

        # Get indices of longitude for every column
        indices_lon=np.zeros((siz_hor[1],2))
        slice_lon=np.zeros((siz_hor[1],))

        for i in range(0,siz_hor[1]): # siz_hor[0] is the number of lines (latitudes), siz_hor[1], the number of columns (longitude)
            idx_lat=np.argmin(np.abs(lat_2D[:,i]-idx))
            indices_lon[i,:]=[idx_lat,i]
            slice_lon[i]=lon_2D[idx_lat,i]
        # Now get data at indices for every vertical layer
        if dim == 3:
            for j in range(0, nVertLayers):
                slice_data[j,:]=[var.data[j,pt[0],pt[1]] for pt in indices_lon]
        elif dim == 2:
            slice_data=np.asarray([var.data[pt[0],pt[1]] for pt in indices_lon])

        if slice_z:
            zlevels=np.zeros((nVertLayers, siz_hor[1]))
            for j in range(0, nVertLayers):
                zlevels[j,:]=[var.attributes['z-levels'][j,pt[0],pt[1]] for pt in indices_lon]
            slice.attributes['z-levels']=zlevels            

        slice.dim = slice.dim - 1
        slice.data=slice_data
        slice.coordinates.pop('lat_2D',None)
        slice.coordinates.pop('lon_2D',None)
        slice.coordinates['lon_slice']=slice_lon
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already have the coordinates in the coordinates field
        slice.name+='_lat_SLICE'
        
    elif type == 'lon':
        
        if idx<var.attributes['domain_2D'][0][1] or idx>var.attributes['domain_2D'][1][1]:
            print 'Specified slicing longitude is outside of model domain!'                
            closest=binary_search(coords,cut_val)
            if len(closest)==1:
                data_interp=var.data[closest,:,:]
            else:
                dist=coords[closest[0]]-coords[closest[1]]
                data_interp=var.data[closest[1],:,:]+(var.data[closest[0],:,:]-var.data[closest[1],:,:])/dist*(cut_val-coords[closest[1]])
            return
            
        try:
            lat_2D=var.coordinates['lat_2D']
        except:
            lat_2D=var.coordinates['lat_2D_resized']

        try:
            lon_2D=var.coordinates['lon_2D']
        except:
            lon_2D=var.coordinates['lon_2D_resized']
            
        siz=var.data.shape
        if dim == 3:
            siz_hor=siz[1:3] # We're not interested in the vertical levels yet
            nVertLayers=siz[0]
        else: # variable is 2D
            siz_hor=siz
            nVertLayers=1

        slice_data=np.zeros((nVertLayers, siz_hor[0]))

        # Get indices of latitudes for every column
        indices_lat=np.zeros((siz_hor[0],2))
        slice_lat=np.zeros((siz_hor[0],))

        for i in range(0,siz_hor[0]): # siz_hor[0] is the number of lines (latitudes), siz_hor[1], the number of columns (longitude)
            idx_lon=np.argmin(np.abs(lon_2D[:,i]-idx))
            indices_lat[i,:]=[i,idx_lon]
            slice_lat[i]=lat_2D[i,idx_lon]
        # Now get data at indices for every vertical layer
        if dim == 3:
            for j in range(0, nVertLayers):
                slice_data[j,:]=[var.data[j,pt[0],pt[1]] for pt in indices_lat]
        elif dim == 2:
            slice_data=np.asarray([var.data[pt[0],pt[1]] for pt in indices_lat])

        if slice_z:
            zlevels=np.zeros((nVertLayers, siz_hor[0]))
            for j in range(0, nVertLayers):
                zlevels[j,:]=[var.attributes['z-levels'][j,pt[0],pt[1]] for pt in indices_lat]
            slice.attributes['z-levels']=zlevels
                
        
        slice.dim = slice.dim - 1
        slice.data=slice_data
        slice.coordinates.pop('lat_2D',None)
        slice.coordinates.pop('lon_2D',None)
        slice.coordinates['lat_slice']=slice_lat
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already have the coordinates in the coordinates field
        slice.name+='_lon_SLICE'

    elif type == 'latlon':
        
        idx=np.asarray(idx)
        try:
            lim_lat=[var.attributes['domain_2D'][0][0],var.attributes['domain_2D'][1][0]]
            lim_lon=[var.attributes['domain_2D'][0][1],var.attributes['domain_2D'][1][1]]
            bad=False
            if idx[0,0]>lim_lat[1] or idx[0,0]<lim_lat[0]:
                bad=True
            elif idx[-1,0]>lim_lat[1] or idx[-1,0]<lim_lat[0]:
                bad=True
                
            elif idx[0,1]>lim_lon[1] or idx[0,1]<lim_lon[0]:
                bad=True
            elif idx[-1,1]>lim_lon[1] or idx[-1,1]<lim_lon[0]:
                bad=True
            if bad:
                print 'Specified slicing transect is outside of model domain!'
                return
        except:
            print 'Specified index does not have the right dimensions!'
            print 'Should be a Nx2 array of lat/lon coordinates!'
            return
            
        try:
            lat_2D=var.coordinates['lat_2D']
        except:
            lat_2D=var.coordinates['lat_2D_resized']
            
        lat_2D_stack=lat_2D.ravel()

        try:
            lon_2D=var.coordinates['lon_2D']
        except:
            lon_2D=var.coordinates['lon_2D_resized']
            
        lon_2D_stack=lon_2D.ravel()

        siz=var.data.shape
        if dim == 3:
            nVertLayers=siz[0]
        else: # variable is 2D
            nVertLayers=1

        n_pts=len(idx)

        combined_latlon = np.dstack([lat_2D_stack,lon_2D_stack])[0]
        slice_data=np.zeros((nVertLayers,n_pts))
        
        # Get indices with kd-tree
        tree = spatial.cKDTree(combined_latlon)
        dist, indexes = tree.query(idx)

        k=0
        
        slice_lat=lat_2D_stack[indexes]
        slice_lon=lon_2D_stack[indexes]
        
        # Get data for every vertical layer
        if dim == 3:
            for j in range(0, nVertLayers):
                data_stack=var.data[j,:,:].ravel()
                slice_data[j,:]=data_stack[indexes]
        elif dim == 2:
            data_stack=var.data.ravel()
            slice_data=data_stack[indexes]
        
        if slice_z:
            zlevels=np.zeros((nVertLayers,n_pts))
            for j in range(0, nVertLayers):
                z_stack=var.attributes['z-levels'][j,:,:].ravel()
                zlevels[j,:]=z_stack[indexes]
        slice.attributes['z-levels']=zlevels    
        
        g = pyproj.Geod(ellps='clrk66') # Use Clarke 1966 ellipsoid.
        
        az12,az21,dist = g.inv(slice_lon[0],slice_lat[0],slice_lon[-1],slice_lat[-1]) # Backward transform
        
        slice.dim = slice.dim - 1
        slice.data=slice_data
        slice.coordinates.pop('lon_2D',None)
        slice.coordinates.pop('lat_2D',None)
        slice.attributes['lat_slice']=slice_lat
        slice.attributes['lon_slice']=slice_lon
        slice.coordinates['distance']=np.linspace(start=0,stop=dist,num=n_pts)
        slice.name+='_k_SLICE'
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
    
    elif type == 'PPI':
                    
        if slice.dim != 3:
            print 'In order to slice on a radar PPI the variable must be 3D'
            print 'Aborting...'
            return None
            
        if not slice_z:
            print 'In order to slice on a radar PPI you first have to assign heights to the COSMO variable using the assign_heights function'
            print 'Aborting...'
            return None
            
        try:
            # Read fields from the input dictionary
            coords_0=idx['radar_pos']
            elevation=idx['elevation']
            maxrange=idx['range']
            resolution=idx['resolution']
        except:
            print 'In order to interpolate COSMO data along a radar PPI, you have to give as second input a dictionary with fields'
            print "'radar_pos' : (lat,lon, altitude) 3D coordinates of the radar in WGS84 coordinates"
            print "'elevation' : double specifying the elevation angle of the PPI scan"
            print "'range' : double or int specifying the maximal PPI range"
            print "'resolution' : the size in meters of the Cartesian grid of interpolation (ex 250 m)"
            return None

            
        # Start by defining the Cartesian grid
        X,Y=np.meshgrid(np.arange(-maxrange,maxrange+resolution,resolution),np.arange(-maxrange,maxrange+resolution,resolution))
        size_PPI=X.shape
        
        # Calculate range of every grid point
        R=np.sqrt(X**2+Y**2)
        Azi=(np.arctan2(-X, -Y)+np.pi)
        
        # Now calculate height and arc distance using 4/3 earth's radius model for refraction
        ke=4/3
        
        elev_rad=np.deg2rad(elevation)
    
        # Compute earth radius at radar latitude 
        EarthRadius=np.sqrt(((a**2*np.cos(coords_0[0]))**2+(b**2*np.sin(coords_0[0]))**2)/((a*np.cos(coords_0[0]))**2+(b*np.sin(coords_0[0]))**2))
        # Compute height over radar of every pixel        
        H=np.sqrt(R**2 + (ke*EarthRadius)**2+2*R*ke*EarthRadius*np.sin(elev_rad))-ke*EarthRadius

        # Compute arc distance of every pixel
        S=ke*EarthRadius*np.arcsin((R*np.cos(elev_rad))/(ke*EarthRadius+H))
        
        # Now using S get the corresponding latitudes and longitudes
        ZoneNumber = np.floor((coords_0[1] + 180)/6) + 1
        p = pyproj.Proj(proj='utm',zone=ZoneNumber,ellps='WGS84')

        x_0,y_0=p(coords_0[1],coords_0[0])

        lons_PPI,lats_PPI = p(x_0+(S*np.sin(Azi)).ravel(),y_0+(S*np.cos(Azi)).ravel(), inverse=True)
        
        try:
            lats_COSMO=var.coordinates['lat_2D']
        except:
            lats_COSMO=var.coordinates['lat_2D_resized']
        lats_COSMO=lats_COSMO.ravel()

        try:
            lons_COSMO=var.coordinates['lon_2D']
        except:
            lons_COSMO=var.coordinates['lon_2D_resized']
            
        lons_COSMO=lons_COSMO.ravel()
        combined_latlon =  np.vstack((lats_COSMO.ravel(),lons_COSMO.ravel())).T

        # Get indices with kd-tree
        tree = spatial.cKDTree(combined_latlon)
        dist, indexes_model = tree.query(np.vstack((lats_PPI,lons_PPI)).T)
        
        H_stack=H.ravel()
        R_stack=R.ravel()
        
        indexes_slice=np.where(R_stack<maxrange)[0]
    
        data_interp=np.zeros((len(indexes_model),))*float('nan')

        size_model_2D=var.data[0,:,:].shape
        list_closest=[]
        delta_dist_list=[]
        for idx_slice in indexes_slice:
            idx_model=indexes_model[idx_slice]
            idx_model_2D=np.unravel_index(idx_model,size_model_2D)
            altitudes_model=var.attributes['z-levels'][:,idx_model_2D[0],idx_model_2D[1]]
            data_model=var.data[:,idx_model_2D[0],idx_model_2D[1]]

            altitude_PPI=H_stack[idx_slice]+coords_0[2] # Add altitude of radar to height above radar
            closest=binary_search(altitudes_model,altitude_PPI)
            list_closest.append(closest)
            try:
                if len(closest)==1:
                    data_interp[idx]=data_model[closest]
                else:
                    delta_dist=(altitude_PPI-altitudes_model[closest[1]])/(altitudes_model[closest[0]]-altitudes_model[closest[1]])
                    delta_dist_list.append(delta_dist)
                    data_interp[idx_slice]=data_model[closest[1]]+(data_model[closest[0]]-data_model[closest[1]])*delta_dist
            except:
                pass
         
        data_PPI=np.reshape(data_interp, size_PPI)
        lons_PPI=np.reshape(lons_PPI,size_PPI)
        lats_PPI=np.reshape(lats_PPI, size_PPI)
        
        # create fast_reslice dictionary
        fast_reslice={}
        fast_reslice['idx_slice']=indexes_slice
        fast_reslice['list_closest']=list_closest
        fast_reslice['idx_model']=indexes_model
        fast_reslice['delta_dist_list']=delta_dist_list
        
        slice.dim = slice.dim
        slice.data = data_PPI
        slice.coordinates.pop('lon_2D',None)
        slice.coordinates.pop('lat_2D',None)
        slice.coordinates['lon_2D_PPI']=lons_PPI
        slice.coordinates['lat_2D_PPI']=lats_PPI
        slice.attributes['altitudes']=H+coords_0[2]
        slice.name+='_PPI_SLICE'
        slice.coordinates.pop('hyb_levels',None) # This key is not needed anymore
        slice.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
        slice.attributes.pop('z-levels',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
        slice.fast_reslice=fast_reslice
        
        
    slice.slice_type=type # Boolean to indicate if variable is a slice
    slice.attributes['slice_params']=type+' = '+str(idx) # Add parameters of the slicing as attributes
    return slice
    
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
    T=pc.get_variable(file_h,'TWC')
    T.assign_heights()
    index_PPI={'range':50000, 'radar_pos':[46.36,7.22,3000],'resolution':500, 'elevation': 1}
    
    start_time = time.time()
    T_slice=slicer(T,'PPI',index_PPI)
    