# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:58:00 2015

@author: wolfensb
"""
import pyproj as pyproj
import numpy as np
import scipy.spatial as spatial

from pycosmo.utilities import WGS_to_COSMO
from pycosmo.beam_tracing import _Beam, refraction_sh, quad_pts_weights, integrate_quad
from pycosmo.c.radar_interp_c import get_all_radar_pts

# Two extreme earth radius
RADIUS_MAX = 6378.1370*1000
RADIUS_MIN = 6356.7523*1000
        
        
def extract(variables, slice_type, idx, parallel = False):
    # Type can be either:
        # "i" (scalar) : Cut at fixed zero-based i-index (no interpolation).
        # "j" (scalar) : Cut at fixed zero-based j-index (no interpolation).
        # "k" (scalar) : Cut at fixed zero-based k-index (no interpolation).
        # "level" (scalar) : Cut at fixed level (e.g. pressure in a p-file or height in a z-file or h-file) (interpolation)
        # "lat" (scalar) : Cut at fixed geographical latitude (interpolation)
        # "lon" (scalar) : Cut at fixed geographical longitude (interpolation)
        # "latlon" (nx2 array) : Cut along list of geographical latitude/longitude pairs (interpolation)
    
    if not isinstance(variables,list):
        variables =  [variables]
    else:   
        # CHECK 1: all variables have same dimensions
        all_shapes = [v[:].shape for v in variables]
        all_domains_llc = [tuple(v.attributes['domain_2D'][0]) for v in variables]
        all_domains_urc = [tuple(v.attributes['domain_2D'][1]) for v in variables]
        shapes_equal = len( set( all_shapes ) ) == 1 
        domains_equal = len( set( all_domains_llc ) ) == 1 and \
                        len( set( all_domains_urc ) )
    
        if not shapes_equal:
            raise ValueError('All variables must have the same dimensions!')
        if not domains_equal:
            raise ValueError('All variables must be defined on the same coordinates')
        
    # CHECK 2: all variables have correct dimensions
    var0 = variables[0]
    # Check dimension of variable
    dim = var0.dim
    # Check if variable has z-levels and if we should slice them:
    slice_z = False
    if 'z-levels' in var0.attributes.keys() and dim == 3:
        slice_z=True
        
    if dim == 2 and (slice_type == 'level' or slice_type == 'k'):
        raise ValueError('Variable must be 3D to cut at vertical coordinates!')
    if (var0.file.type=='h' and 'z-levels' not in var0.attributes.keys()) and slice_type == 'level':
        raise ValueError('In order to cut at a fixed height for hybrid layer files'+
                         'you need to first assign altitude levels using the assign_heights function...')

    slices = [] # initialize list of slices
    slices = [var.copy() for var in variables]
              
    if slice_type == 'i': # Cut at fixed zero-based i-index (no interpolation)
        for i,var in enumerate(variables):
            if dim == 2: slices[i].data=var[:][idx,:]
            elif dim == 3: slices[i].data=var[:][:,idx,:]
            # Assign new coordinates
            if slice_z:
                slices[i].attributes.pop('z-levels',None)
                slices[i].attributes['z-levels'] = var.attributes['z-levels'][:,idx,:]
            slices[i].coordinates['lat_2D']=var.coordinates['lat_2D'][idx]
            slices[i].dim = slices[i].dim - 1
            slices[i].attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
            slices[i].name+='_i_SLICE'

    elif slice_type == 'j': #  Cut at fixed zero-based j-index (no interpolation)
        for i,var in enumerate(variables):
            if dim == 2: slices[i].data=var[:][:,idx]
            elif dim == 3: slices[i].data=var[:][:,:,idx]
            if slice_z:
                slices[i].attributes.pop('z-levels',None)
                slices[i].attributes['z-levels']=var.attributes['z-levels'][:,:,idx]
            slices[i].coordinates['lon_2D']=var.coordinates['lon_2D'][idx]
            slices[i].dim = slice.dim - 1
            slices[i].attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field         
            slices[i].name+='_j_SLICE'
            
    elif slice_type == 'k': #  Cut at fixed zero-based k-index (no interpolation)
        for i,var in enumerate(variables):
            slices[i].data=var[:][idx,:,:]
            try:
                slices[i].coordinates['heights']=var.coordinates['heights'][idx,:,:]
            except: # variable has not been georeferenced vertically...
                 slices[i].coordinates['hyb_levels']=var.coordinates['hyb_levels'][idx]
            if slice_z:
                slices[i].attributes['z-levels']=var.attributes['z-levels'][idx,:,:]
                
            slices[i].name+='_k_SLICE'
            slices[i].attributes.pop('domain_2D',None) # This key is not needed anymore since we have already the coordinates in the coordinates field
            slices[i].dim = slice.dim - 1
            
    elif slice_type == 'level':
        var = variables[0] # Start with first variable
        
        cut_val=idx
        siz=var[:].shape
        if var.file.type == 'p' or var.file.type == 'z':
            if var.file.slice_type == 'p':
                print('File is a p-file, trying to cut at p='+str(idx)+'hPa...')
                coord_name='press_levels'
            elif var.file.slice_type == 'z' :
                print('File is a z-file, trying to cut at z='+str(idx)+'m...')
                coord_name='height_levels'
    
            coords=var.coordinates[coord_name]
            if not cut_val in coords: # Linear interpolation
                closest=pc.binary_search(coords,cut_val)
                
                for i,var in enumerate(variables):
                    if len(closest)==1:
                        slices[i].data = var[:][closest,:,:]
                    else:
                        dist=coords[closest[0]]-coords[closest[1]]
                        slices[i].data = var[:][closest[1],:,:]+(var[:][closest[0],:,:]-\
                            var[:][closest[1],:,:])/dist*(cut_val-coords[closest[1]])     
            else:
                idx_cut=np.where(coords == cut_val)[0]
                for i,var in enumerate(variables):
                    slices[i].data = var[:][idx_cut[0],:,:]
            
        elif var.file.type == 'h':
            coord_name='hyb_levels'
        
            zlevels = var.attributes['z-levels']
            arr = (zlevels-cut_val)
            arr[arr<0] = 99999 # Do this in order to find only the indexes of heights > cut_val
            
            indexes_larger = np.argmin(arr,axis=0)
            indexes_smaller = indexes_larger+1 # remember that array is in decreasing order...
            indexes_smaller[indexes_smaller>siz[0]-1] = siz[0]-1
            k,j = np.meshgrid(np.arange(siz[2]),np.arange(siz[1])) # One of numpy's limitation is that we have to index with these index arrays (see next line)
            dist=zlevels[indexes_larger,j,k]-zlevels[indexes_smaller,j,k]
            
            for i,var in enumerate(variables):
                # Perform linear interpolation
                data_interp=var[:][indexes_smaller,j,k]+(var[:][indexes_larger,j,k]-\
                    var[:][indexes_smaller,j,k])/dist*(cut_val-zlevels[indexes_smaller,j,k])
           
                # Now mask
                data_interp[indexes_smaller==siz[0]-1]=float('nan')
                
                slices[i].data = data_interp
                
        for sli in slices:
            sli.attributes.pop(coord_name,None)
            sli.dim = sli.dim - 1
            sli.name+='_level_SLICE'

    elif slice_type == 'lat':

        # First we check if the given profile is valid
        if idx<var0.attributes['domain_2D'][0][0] or idx>var0.attributes['domain_2D'][1][0]:
            
            raise ValueError('Specified slicing longitude is outside of model domain!'+
            'Model domain is: '+str(var.attributes['domain_2D']))
            
            return
            
        # Get latitudes and longitudes
        try:
            lat_2D=var0.coordinates['lat_2D']
        except:
            lat_2D=var0.coordinates['lat_2D_resized']
        
        try:
            lon_2D=var0.coordinates['lon_2D']
        except:
            lon_2D=var0.coordinates['lon_2D_resized']
        
        # Get dimensions
        siz=var0[:].shape
        if dim == 3:
            n_lon = siz[2]
            n_vert = siz[0]
        else: # variable is 2D
            n_lon = siz[1]
            n_vert = 1

        # Get indices of longitude for every column
        # Initialize
        indices_lon = np.zeros((n_lon,2),dtype=int) 
        slice_lon = np.zeros((n_lon,))
        for i in range(0,n_lon): # Loop on all longitudes
            # Get the idx for the specified latitude
            idx_lat = int(np.argmin(np.abs(lat_2D[:,i]-idx)))
            indices_lon[i,:] = [idx_lat,i]
            slice_lon[i] = lon_2D[idx_lat,i]
        # Now get data at indices for every vertical layer  
        for i,var in enumerate(variables):
            # Initialize interpolated values
            slices[i].data = np.zeros((n_vert,n_lon))
            if dim == 3:
                for j in range(0, n_vert):
                    slices[i].data[j,:] = [var[:][j,pt[0],pt[1]] for pt in indices_lon]
            elif dim == 2:
                slices[i].data = np.asarray([var[:][pt[0],pt[1]] for pt in indices_lon])
        
        # If variable has z-levels, get them along profile
        if slice_z:
            zlevels=np.zeros((n_vert, n_lon))
            for j in range(0, n_vert):
                zlevels[j,:]=[var0.attributes['z-levels'][j,pt[0],pt[1]] for pt in indices_lon]
                
        for sli in slices:
            # If variable has z-levels, get them along profile
            if slice_z: sli.attributes['z-levels']=zlevels            
    
            sli.dim = sli.dim - 1
            sli.coordinates.pop('lat_2D',None)
            sli.coordinates.pop('lon_2D',None)
            sli.coordinates['lon_slice'] = slice_lon
            sli.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already have the coordinates in the coordinates field
            sli.name+='_lat_SLICE'
        
    elif slice_type == 'lon':
        
        # First we check if the given profile is valid
        if idx<var0.attributes['domain_2D'][0][1] or idx>var0.attributes['domain_2D'][1][1]:
            
            raise ValueError('Specified slicing longitude is outside of model domain!'+
            'Model domain is: '+str(var.attributes['domain_2D']))
            
            return

        # Get latitudes and longitudes            
        try:
            lat_2D = var0.coordinates['lat_2D']
        except:
            lat_2D = var0.coordinates['lat_2D_resized']

        try:
            lon_2D = var0.coordinates['lon_2D']
        except:
            lon_2D = var0.coordinates['lon_2D_resized']
            
            
        # Get dimensions
        siz = var0[:].shape
        if dim == 3:
            n_lat = siz[1]
            n_vert = siz[0]
        else: # variable is 2D
            n_lat = siz[0]
            n_vert = 1
            
        # Get indices of latitudes for every column
        # Initialize
        indices_lat = np.zeros((n_lat,2),dtype=int)
        slice_lat = np.zeros((n_lat,))

        for i in range(0,n_lat): # Loop on all latitudes
            idx_lon = np.argmin(np.abs(lon_2D[i,:]-idx))
            indices_lat[i,:] = [i,idx_lon]
            slice_lat[i] = lat_2D[i,idx_lon]

        # Now get data at indices for every vertical layer  
        for i,var in enumerate(variables):
            # Initialize interpolated values
            slices[i].data = np.zeros((n_vert,n_lat))
            if dim == 3:
                for j in range(0, n_vert):
                    slices[i].data[j,:] = [var[:][j,pt[0],pt[1]] for pt in indices_lat]
            elif dim == 2:
                slices[i].data = np.asarray([var[:][pt[0],pt[1]] for pt in indices_lat])
       
        # If variable has z-levels, get them along profile
        if slice_z:
            zlevels=np.zeros((n_vert, n_lat))
            for j in range(0, n_vert):
                zlevels[j,:]=[var0.attributes['z-levels'][j,pt[0],pt[1]] for pt in indices_lat]
                
        for sli in slices:
            # If variable has z-levels, get them along profile
            if slice_z: sli.attributes['z-levels']=zlevels            
    
            sli.dim = sli.dim - 1
            sli.coordinates.pop('lat_2D',None)
            sli.coordinates.pop('lon_2D',None)
            sli.coordinates['lat_slice'] = slice_lat
            sli.attributes.pop('domain_2D',None) # This key is not needed anymore since we have already have the coordinates in the coordinates field
            sli.name+='_lon_SLICE'
            
    elif slice_type == 'latlon':
        # First we check if the given profile is valid
        if len(idx.shape) !=2 or idx.shape[1]!=2:
            raise ValueError('Specified index does not have the right dimensions!'+
            'Should be a Nx2 array of lat/lon coordinates!'+
            'Please make sure to generate the profile with the coords_profile function')
            
        # Now we check if all points of the profile are within domain
        lim_lat = [var0.attributes['domain_2D'][0][0],var0.attributes['domain_2D'][1][0]]
        lim_lon = [var0.attributes['domain_2D'][0][1],var0.attributes['domain_2D'][1][1]]
        flag_bad = False # Flag
        
        # Check if beg. and end of profile are within domain
        if idx[0,0] > lim_lat[1] or idx[0,0]<lim_lat[0]:
            flag_bad=True
        elif idx[-1,0] > lim_lat[1] or idx[-1,0]<lim_lat[0]:
            flag_bad=True
        elif idx[0,1] > lim_lon[1] or idx[0,1]<lim_lon[0]:
            flag_bad=True
        elif idx[-1,1] > lim_lon[1] or idx[-1,1]<lim_lon[0]:
            flag_bad=True
        if flag_bad:
            raise ValueError('Some points of profile are outside COSMO domain!',    
                  'Model domain is: '+str(var.attributes['domain_2D']))
            return
  
        # Get latitudes and longitudes        
        try:
            lat_2D = var0.coordinates['lat_2D']
        except:
            lat_2D = var0.coordinates['lat_2D_resized']
            
        lat_2D_stack=lat_2D.ravel()

        try:
            lon_2D = var0.coordinates['lon_2D']
        except:
            lon_2D = var0.coordinates['lon_2D_resized']
            
        lon_2D_stack = lon_2D.ravel()

        # Get dimensions
        siz = var0[:].shape
        if dim == 3:
            n_vert = siz[0]
        else: # variable is 2D
            n_vert = 1

        n_pts = len(idx)

        # Get array of all latitudes and longitudes of COSMO grid points
        combined_latlon = np.dstack([lat_2D_stack,lon_2D_stack])[0]
        
        # Create kd-tree of all COSMO points
        tree = spatial.cKDTree(combined_latlon)
        # Now get indexes of profile points
        dist, indexes = tree.query(idx)
        
        # Get latitude and longitudes corresponding to indexed
        slice_lat = lat_2D_stack[indexes]
        slice_lon = lon_2D_stack[indexes]

        # Now get data at indices for every vertical layer  
        for i,var in enumerate(variables):
            # Initialize interpolated values 
            slices[i].data = np.zeros((n_vert,n_pts))
            if dim == 3: # 3D case
                for j in range(0, n_vert):
                    data_stack = var[:][j,:,:].ravel()
                    slices[i].data[j,:] = data_stack[indexes]
            elif dim == 2: # 2D case (trivial, vert. col. is 1 pt only)
                data_stack=var[:].ravel()
                slices[i].data = data_stack[indexes]

        # Now we get the distance along the profile as the new coordinates
        g = pyproj.Geod(ellps='WGS84') # Use WGS84 ellipsoid.
        az12,az21,dist = g.inv(slice_lon[0],slice_lat[0],slice_lon[-1],slice_lat[-1]) # Backward transform

        # If variable has z-levels, get them along profile
        if slice_z:
            zlevels=np.zeros((n_vert,n_pts))
            for j in range(0, n_vert):
                z_stack = var0.attributes['z-levels'][j,:,:].ravel()
                zlevels[j,:] = z_stack[indexes]
                
        for sli in slices:
            # If variable has z-levels, get them along profile
            if slice_z: sli.attributes['z-levels'] = zlevels            
    
            sli.dim = sli.dim - 1
            sli.coordinates.pop('lon_2D',None)
            sli.coordinates.pop('lat_2D',None)
            sli.attributes['lat_slice']=slice_lat
            sli.attributes['lon_slice']=slice_lon
            sli.coordinates['distance']=np.linspace(start=0,stop=dist,num=n_pts)
            sli.name+='_latlon_SLICE'
            sli.attributes.pop('domain_2D',None) # This key is not needed anymore

    elif slice_type == 'PPI':
        # First we check if the given variables are compatible with a PPI scan                    
        if var0.dim != 3:
            raise ValueError('In order to slice on a radar RHI the variable '+
                             'must be 3D. Aborting...')            
        if not slice_z:
            raise ValueError('In order to slice on a radar RHI you first have '+
                             'to assign heights to the COSMO variable using '+
                             'the assign_heights function')
                             
        # Check scan specifications
        # Mandatory arguments
        try:
            # Read fields from the input dictionary
            rpos = [float(i) for i in idx['rpos']] # radar coords.
            elevation = float(idx['elevation']) # elevation angle  
            maxrange = float(idx['maxrange']) # maximum range
            rresolution = float(idx['rresolution']) # radial resolution
            beamwidth_3dB = float(idx['beamwidth']) # 3dB antenna beamwidth
        except:
            raise ValueError('In order to interpolate COSMO data along a radar'+
                 'RHI, you have to give as second input a dictionary with '+
                 'fields \n rpos : (lat,lon, altitude) 3D coordinates of'+
                 ' the radar in WGS84 coordinates \n elevation : double '+
                 ' specifying the elevation angle of the PPI scan \n maxrange : '+
                 ' double or int specifying the maximal PPI range \n'+
                 'rresolution : the radial resolution of the radar')           
        # Optional arguments:
        try:
            azimuths = np.array(idx['azimuths'])
        except:
            print("Could not read 'azimuth' field in input dict."+\
                  'Using default one: np.arange(0,360,beamwidth_3dB)')
            azimuths = np.arange(0,360+beamwidth_3dB,beamwidth_3dB)
        
        try:
            npts_quad = [int(i) for i in idx['npts_quad']]
        except:
            print("Could not read 'npts_quad' field in input dict."+\
                  'Using default one: [5,3]')
            npts_quad = [5,5]

        try:
            refraction_method = int(idx['refraction_method'])
        except:
            print("Could not read 'refraction_method' field in input dict."+\
                  'Using default one: 1')
            refraction_method = 1
        
        # If refraction_method == 2: try to get N from options
        N = []
        if refraction_method == 2:
            try: 
                print('Trying to compute atm. refractivity (N) from COSMO file')
                cfile = var0.attributes['c-file']
                N = var0.file.get_variable('N',assign_heights = True,
                                           cfile_name=cfile)
            except:
                pass
                refraction_method = 1
                print('Could not compute N from COSMO data'+
                      ' Using refraction_method = 1, instead')
        # Get the quadrature points        
        pts,weights = quad_pts_weights(beamwidth_3dB, npts_quad)
        
        # Get radar gates distances
        rangevec = np.arange(0,maxrange,rresolution)

        # Get model heights and COSMO proj from first variable    
        model_heights = var0.attributes['z-levels']
        rad_interp_values = np.zeros(len(rangevec),)*float('nan')
        
        # Get COSMO local coordinates info
        proj_COSMO = var0.attributes['proj_info']
        
        # Get lower left corner of COSMO domain in local coordinates
        llc_COSMO = (float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
        llc_COSMO = np.asarray(llc_COSMO).astype('float32')
        
        # Get upper left corner of COSMO domain in local coordinates
        urc_COSMO = (float(proj_COSMO['Lo2']), float(proj_COSMO['La2']))
        urc_COSMO = np.asarray(urc_COSMO).astype('float32')

        # Get resolution             
        res_COSMO = var0.attributes['resolution']
                    
        # Initialize WGS84 geoid
        g = pyproj.Geod(ellps='WGS84')
        
        # Initialize interpolated values and coords
        for i,var in enumerate(variables):
            slices[i].data = np.zeros((len(rangevec),len(azimuths)))
            
        lons_scan =  np.zeros((len(rangevec),len(azimuths)))
        lats_scan = np.zeros((len(rangevec),len(azimuths)))
        heights_scan = np.zeros((len(rangevec),len(azimuths)))
        dist_ground = np.zeros((len(rangevec),len(azimuths)))
        frac_power = np.zeros((len(rangevec),len(azimuths)))
        
        # Main loop
        for radial,az in enumerate(azimuths): # Loop on all radials
            print('Processing azimuth ang. '+str(az))
            list_beams = []
            for j,pt_vert in enumerate(pts[1]): 
                s,h = refraction_sh(rangevec,pt_vert+elevation,rpos,
                                    refraction_method, N)
                for i,pt_hor in enumerate(pts[0]):
                    # Get radar gate coordinates in WGS84
                    lons_radial = []
                    lats_radial = []
                    for s_gate in s:
                        # Note that pyproj uses lon, lat whereas I use lat, lon
                        lon,lat,ang = g.fwd(rpos[1],rpos[0], pt_hor + az, s_gate)  
                        lons_radial.append(lon)
                        lats_radial.append(lat)

                    # Transform radar gate coordinates into local COSMO coordinates
                    coords_rad_loc= WGS_to_COSMO((lats_radial,lons_radial),
                                          [proj_COSMO['Latitude_of_southern_pole'],\
                                           proj_COSMO['Longitude_of_southern_pole']])  

                    # Check if all points are within COSMO domain
                    
                    if np.any(coords_rad_loc[:,1]<llc_COSMO[0]) or\
                        np.any(coords_rad_loc[:,0]<llc_COSMO[1]) or \
                            np.any(coords_rad_loc[:,1]>urc_COSMO[0]) or \
                                np.any(coords_rad_loc[:,0]>urc_COSMO[1]):
                                    raise(IndexError('Radar domain is not entirely contained'+
                                                     ' in COSMO domain, aborting'))
                    dic_beams = {}
                    for k,var in enumerate(variables):           
                        model_data=var[:]
                        rad_interp_values = get_all_radar_pts(len(rangevec),\
                                          coords_rad_loc,h,model_data, \
                                          model_heights, llc_COSMO,res_COSMO)[1]
            
                        if k == 0: # Do this only for the first variable (same mask for all variables)
                            mask_beam = np.zeros((len(rad_interp_values)))
                            mask_beam[rad_interp_values == -9999] = -1 # Means that the interpolated point is above COSMO domain
                            mask_beam[np.isnan(rad_interp_values)] = 1  # Means that the interpolated point is below COSMO terrain
                        rad_interp_values[mask_beam!=0] = float('nan') # Assign NaN to all missing data
                        dic_beams[var.name] = rad_interp_values
                        
                        list_beams.append(_Beam(dic_beams, mask_beam, lats_radial,
                            lons_radial, s,h,[pt_hor + az,pt_vert+elevation],
                            weights[i,j]))         
                        
            # Integrate all sub-beams
            scan, frac_pow = integrate_quad(list_beams)    
            
            # Add radial to slices
            for sli in slices: 
                sli[:][:,radial] = scan.values[sli.name]
            # Add radial to coords
  
            lats_scan[:,radial] = scan.lats_profile
            lons_scan[:,radial] = scan.lons_profile
            heights_scan[:,radial] = scan.heights_profile
            dist_ground[:,radial] = scan.dist_profile
            frac_power[:,radial] = frac_pow
            
        # Final bookkeeping
        for sli in slices:      
            sli.dim = sli.dim - 1
            sli.coordinates.pop('lon_2D',None)
            sli.coordinates.pop('lat_2D',None)
            sli.coordinates['range'] = rangevec
            sli.coordinates['azimuth'] = azimuths
            sli.attributes['lon_2D'] = lons_scan
            sli.attributes['lat_2D'] = lats_scan
            sli.attributes['altitude'] = heights_scan
            sli.attributes['elevation'] = elevation
            sli.attributes['fraction_power'] = frac_power
            sli.attributes['dist_ground'] = dist_ground
            sli.name+='_PPI_SLICE'
            sli.coordinates.pop('hyb_levels',None) # not needed anymore
            sli.attributes.pop('domain_2D',None) # not needed anymore 
            sli.attributes.pop('z-levels',None) # not needed anymore 
                
    elif slice_type == 'RHI':
        # First we check if the given variables are compatible with a PPI scan                    
        if var0.dim != 3:
            raise ValueError('In order to slice on a radar RHI the variable '+
                             'must be 3D. Aborting...')            
        if not slice_z:
            raise ValueError('In order to slice on a radar RHI you first have '+
                             'to assign heights to the COSMO variable using '+
                             'the assign_heights function')
                             
        # Check scan specifications
        # Mandatory arguments
        try:
            # Read fields from the input dictionary
            rpos = [float(i) for i in idx['rpos']] # radar coords.
            azimuth = float(idx['azimuth']) # azimuth angle  
            maxrange = float(idx['maxrange']) # maximum range
            rresolution = float(idx['rresolution']) # radial resolution
            beamwidth_3dB = float(idx['beamwidth']) # 3dB antenna beamwidth
        except:
            raise ValueError('In order to interpolate COSMO data along a radar '+
                 'RHI, you have to give as second input a dictionary with '+
                 'fields \n rpos : (lat,lon, altitude) 3D coordinates of'+
                 ' the radar in WGS84 coordinates \n elevation : double '+
                 ' specifying the elevation angle of the PPI scan \n maxrange : '+
                 ' double or int specifying the maximal PPI range \n'+
                 'rresolution : the radial resolution of the radar')           
        # Optional arguments:
        try:
            elevations = np.array(idx['elevation'])
        except:
            print("Could not read 'elevation' field in input dict."+\
                  'Using default one: np.arange(0,360,beamwidth_3dB)')
            elevations = np.arange(0,180+beamwidth_3dB,beamwidth_3dB)
        
        try:
            npts_quad = [int(i) for i in idx['npts_quad']]
        except:
            print("Could not read 'npts_quad' field in input dict."+\
                  'Using default one: [5,5]')
            npts_quad = [5,5]

        try:
            refraction_method = int(idx['refraction_method'])
        except:
            print("Could not read 'refraction_method' field in input dict."+\
                  'Using default one: 1')
            refraction_method = 1
        
        # If refraction_method == 2: try to get N from options
        N = []
        if refraction_method == 2:
            try: 
                print('Trying to compute atm. refractivity (N) from COSMO file')
                cfile = var0.attributes['c-file']
                N = var0.file.get_variable('N',assign_heights = True,cfile=cfile)
            except:
                refraction_method = 1
                print('Could not compute N from COSMO data'+
                      ' Using refraction_method = 1, instead')
        # Get the quadrature points        
        pts,weights = quad_pts_weights(beamwidth_3dB, npts_quad)
        
        # Get radar gates distances
        rangevec = np.arange(0,maxrange,rresolution)

        # Get model heights and COSMO proj from first variable    
        model_heights = var0.attributes['z-levels']
        rad_interp_values = np.zeros(len(rangevec),)*float('nan')
        
        # Get COSMO local coordinates info
        proj_COSMO = var0.attributes['proj_info']
        
        # Get lower left corner of COSMO domain in local coordinates
        llc_COSMO = (float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
        llc_COSMO = np.asarray(llc_COSMO).astype('float32')
        
        # Get upper left corner of COSMO domain in local coordinates
        urc_COSMO = (float(proj_COSMO['Lo2']), float(proj_COSMO['La2']))
        urc_COSMO = np.asarray(urc_COSMO).astype('float32')

        # Get resolution             
        res_COSMO = var0.attributes['resolution']
                    
        # Initialize WGS84 geoid
        g = pyproj.Geod(ellps='WGS84')
        
        # Initialize interpolated values and coords
        for i,var in enumerate(variables):
            slices[i].data = np.zeros((len(rangevec),len(elevations)))
            
        lons_scan =  np.zeros((len(rangevec),len(elevations)))
        lats_scan = np.zeros((len(rangevec),len(elevations)))
        dist_ground = np.zeros((len(rangevec),len(elevations)))
        heights_scan = np.zeros((len(rangevec),len(elevations)))
        frac_power = np.zeros((len(rangevec),len(elevations)))
        
        # Main loop
        for radial,elev in enumerate(elevations): # Loop on all radials
            print('Processing elevation ang. '+str(elev))
            list_beams = []
            for j,pt_vert in enumerate(pts[1]): 
                s,h = refraction_sh(rangevec,pt_vert+elev,rpos,
                                    refraction_method, N)
                for i,pt_hor in enumerate(pts[0]):
                    # Get radar gate coordinates in WGS84
                    lons_radial = []
                    lats_radial = []
                    for s_gate in s:
                        # Note that pyproj uses lon, lat whereas I use lat, lon
                        lon,lat,ang = g.fwd(rpos[1],rpos[0], pt_hor + azimuth,
                                            s_gate)  
                        lons_radial.append(lon)
                        lats_radial.append(lat)

                    # Transform radar gate coordinates into local COSMO coordinates
                    coords_rad_loc= WGS_to_COSMO((lats_radial,lons_radial),
                                  [proj_COSMO['Latitude_of_southern_pole'],\
                                   proj_COSMO['Longitude_of_southern_pole']])  

                    # Check if all points are within COSMO domain
                    
                    if np.any(coords_rad_loc[:,1]<llc_COSMO[0]) or\
                        np.any(coords_rad_loc[:,0]<llc_COSMO[1]) or \
                            np.any(coords_rad_loc[:,1]>urc_COSMO[0]) or \
                                np.any(coords_rad_loc[:,0]>urc_COSMO[1]):
                                    raise(IndexError('Radar domain is not entirely contained'+
                                                     ' in COSMO domain, aborting'))
                    dic_beams = {}
                    for k,var in enumerate(variables):           
                        model_data=var[:]
                        rad_interp_values = get_all_radar_pts(len(rangevec),\
                                          coords_rad_loc,h,model_data, \
                                          model_heights, llc_COSMO,res_COSMO)[1]
            
                        if k == 0: # Do this only for the first variable (same mask for all variables)
                            mask_beam = np.zeros((len(rad_interp_values)))
                            mask_beam[rad_interp_values == -9999] = -1 # Means that the interpolated point is above COSMO domain
                            mask_beam[np.isnan(rad_interp_values)] = 1  # Means that the interpolated point is below COSMO terrain
                        rad_interp_values[mask_beam!=0] = float('nan') # Assign NaN to all missing data
                        dic_beams[var.name] = rad_interp_values
                        
                        list_beams.append(_Beam(dic_beams, mask_beam, lats_radial,
                            lons_radial, s,h,[pt_hor + azimuth,pt_vert+elev],
                            weights[i,j]))         
                        
            # Integrate all sub-beams
            scan, frac_pow = integrate_quad(list_beams)    
            
            # Add radial to slices
            for sli in slices: 
                sli[:][:,radial] = scan.values[sli.name]
            # Add radial to coords
  
            lats_scan[:,radial] = scan.lats_profile
            lons_scan[:,radial] = scan.lons_profile
            heights_scan[:,radial] = scan.heights_profile
            dist_ground[:,radial] = scan.dist_profile
            frac_power[:,radial] = frac_pow
            
        # Final bookkeeping
        for sli in slices:      
            sli.dim = sli.dim - 1
            sli.coordinates.pop('lon_2D',None)
            sli.coordinates.pop('lat_2D',None)
            sli.coordinates['range'] = rangevec
            sli.coordinates['elevation'] = elevations
            sli.attributes['dist_ground'] = dist_ground
            sli.attributes['lon_2D'] = lons_scan
            sli.attributes['lat_2D'] = lats_scan
            sli.attributes['altitude'] = heights_scan
            sli.attributes['frac_power'] = frac_power
            sli.attributes['azimuth'] = azimuth
            sli.name+='_RHI_SLICE'
            sli.coordinates.pop('hyb_levels',None) # not needed anymore
            sli.attributes.pop('domain_2D',None) # not needed anymore 
            sli.attributes.pop('z-levels',None) # not needed anymore 
    else:
        raise ValueError("Invalid slice type, use either 'i','j','k','lat','lon'"+
        "'latlon','PPI' or 'RHI'")
    for sli in slices: 
        sli.slice_type=slice_type # Boolean to indicate if variable is a slice
        sli.attributes['slice_params']=slice_type+' = '+str(idx) # Add parameters of the slicing as attributes
    
    # If there is only one variable, take it out of the list
    if len(slices) == 1:
        slices = slices[0]
        
    return slices

def coords_profile(start, stop, step=-1, npts=-1):
    # This function gets points along a profile_instance specified by a tuple of starting coordinates (lat/lon) 
    # and ending coordinates (lat/lon). Either a number of points can be specified, in which case the profile_instance 
    # will consist of N linearly spaced points or a constant distance step, in which case the number of points in the profile_instance 
    # will be the total distance divided by the distance step.
    # This function is particularly convenient when we want to create a slice with the 'latlon' option
    start=np.asarray(start)
    stop=np.asarray(stop)
    use_step=False
    use_npts=False
    # Check inputs
    if step <= 0 and npts < 3:
        print 'Neither a valid distance step nor a valid number of points of the transect have been specified, please provide one or the other!'
        print 'Number of points must be larger than 3 and step distance must be larger than 0'
        return []
    elif step > 0 and npts >= 3:
        print 'Both a distance step and a number of points in the transect have been specified, only the distance step will be used!'
        use_step=True
    elif step > 0 and npts < 3:
        use_step=True
    else:
        use_npts=True
        
    g = pyproj.Geod(ellps='clrk66') # Use Clarke 1966 ellipsoid.
    az12,az21,dist = g.inv(start[1],start[0],stop[1],stop[0]) # Backward transform
    
    if use_step:
        npts=np.floor(dist/step)
        dist=npts*step
        endlon, endlat, backaz = g.fwd(start[1],start[0],az12, dist)
        profile_instance = g.npts(start[1],start[0], endlon, endlat ,npts-2)
    if use_npts:
        profile_instance = g.npts(start[1],start[0],stop[1],stop[0],npts-2)
    
    # Add start and stop points
    profile_instance.insert(0,(start[1],start[0]))
    profile_instance.insert(len(profile_instance),(stop[1],stop[0]))
    profile_instance=np.asarray(profile_instance)
    # Swap columns to get latitude first (lat/lon)
    profile_instance[:,[0, 1]] = profile_instance[:,[1, 0]]
    
    return profile_instance
