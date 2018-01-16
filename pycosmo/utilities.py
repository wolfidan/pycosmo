 # -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:31:55 2015

@author: wolfensb
"""
import pycosmo.c.interp1_c as interp1_c

from scipy.interpolate import interp1d
from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import glob, os
import re
import datetime



def binary_search(vec, val):
    hi=0
    lo=len(vec)-1
    if(val>vec[0] or val<vec[-1]):
        return -1
    while hi < lo:
        mid = (lo+hi)//2
        midval = vec[mid]
        if val == midval:
            return [mid,mid]
        elif lo == mid or hi == mid:
            return [hi, lo]
        elif midval < val:
            lo = mid
        elif midval > val:
            hi = mid
    return -1

def get_model_filenames(folder):
    filenames={}
    filenames['c']=[]
    filenames['p']=[]
    filenames['h']=[]
    filenames['z']=[]

    filenames_temp=sorted(glob.glob(folder+'/*'))

    for f in filenames_temp:
        fileName, fileExtension = os.path.splitext(f)
        if fileExtension in ['.nc','.cdf','.netcdf','.grib','.grb1','.grb2','','.GRIB']:

            if fileName[-1] == 'c':
                filenames['c'].append(f)
            elif fileName[-1] == 'p':
                filenames['p'].append(f)
            elif fileName[-1] == 'z':
                filenames['z'].append(f)
            else:
                filenames['h'].append(f)

    return filenames

def get_time_from_COSMO_filename(fname, spinup=12):

    bname=os.path.basename(fname)
    dirname=os.path.dirname(fname)
    event_date=re.findall(r"([0-9]{10})",dirname)[0]
    event_date=datetime.datetime.strptime(event_date,'%Y%m%d%H')
    tdelta=datetime.timedelta(days=int(bname[4:6]),hours=int(bname[6:8])-spinup,minutes=int(bname[8:10]),seconds=int(bname[10:12]))
    time=event_date+tdelta
    return time


def move_element(odict, thekey, newpos):
    odict[thekey] = odict.pop(thekey)
    i = 0
    for key, value in odict.items():
        if key != thekey and i >= newpos:
            odict[key] = odict.pop(key)
        i += 1
    return odict

def piecewise_linear(x,y):
    interpolator = interp1d(x,y)
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if np.isscalar(xs):
            xs=[xs]
        return np.array([pointwise(xi) for xi in xs])

    return ufunclike

def resize_domain(var, boundaries):
    # Boundaries format are [[lower_left_lat, lower_left_lon],[upper_right_lat, upper_right_lon]]
    if 'lat_2D' not in var.coordinates.keys() or 'lon_2D' not in var.coordinates.keys():
        print('To resize the domain, the variable must be 2D or 3D and be georeferenced!')
        return
    else:
        cp=var.copy()
        lat_2D=var.coordinates['lat_2D']
        lon_2D=var.coordinates['lon_2D']
        match_lat=np.logical_and(lat_2D>boundaries[0][0], lat_2D<boundaries[1][0])
        match_lon=np.logical_and(lon_2D>boundaries[0][1], lon_2D<boundaries[1][1])

        match_all=match_lon&match_lat == True
        B = np.argwhere(match_all)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

        if var.dim == 2:
            cp.data=var.data[ystart:ystop, xstart:xstop]
        else:
            cp.data=var.data[:,ystart:ystop, xstart:xstop]

        cp.coordinates.pop('lat_2D',None)
        cp.coordinates.pop('lon_2D',None)
        cp.coordinates['lon_2D_resized']=lon_2D[ystart:ystop, xstart:xstop]
        cp.coordinates['lat_2D_resized']=lat_2D[ystart:ystop, xstart:xstop]

        cp.attributes['domain_2D']=boundaries
        return cp


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)
    try:
        print('Triming...command is:')
        print('convert -density '+str(kwargs['dpi'])+' '+args[0]+' -trim +repage ' + args[0])
        subprocess.call('convert -density '+str(kwargs['dpi'])+' '+args[0]+' -trim +repage ' + args[0],shell=True)
    except:
        print('Triming failed, check if imagemagick is installed')
        pass
    return

def savevar(list_vars, name='output.nc'):
    try:
        f = netcdf.NetCDFFile(name, 'w')
    except:
        raise IOError('Could not create or open file_instance '+name)

    if not isinstance(list_vars, list):
        # Only one variable, put it into list
        list_vars=[list_vars]

    for var in list_vars:
        siz = var[:].shape
        list_dim_names = []
        for idx,dim in enumerate(var.dimensions):
            if dim not in f.dimensions.keys():
                f.createDimension(dim, siz[idx])
            list_dim_names.append(dim)
        varhandle = f.createVariable(var.name,'f', tuple(list_dim_names))
        varhandle[:] = var[:]
        coordinates = []
        for coords in var.coordinates.keys():
            if coords not in f.variables.keys():
                if 'lon_2D' in coords:
                    coordname = var.dimensions[-1].replace('y','lon')
                    handle = f.createVariable(coordname,'f',
                           tuple(list_dim_names[-2:]))
                elif 'lat_2D' in coords:
                    coordname = var.dimensions[-2].replace('x','lat')
                    handle = f.createVariable(coordname,'f',
                           tuple(list_dim_names[-2:]))

                else:
                    coordname = list_dim_names[0]
                    handle = f.createVariable(coordname,'f', tuple([list_dim_names[0]]))
                coordinates.append(coordname)
                handle[:] = var.coordinates[coords]

        for attr in var.attributes.keys():
            if attr not in ['z-levels','domain_2D']:
                if isinstance(var.attributes[attr],dict):
                    for key in var.attributes[attr].keys():
                        setattr(varhandle, key, var.attributes[attr][key])
                else:
                    setattr(varhandle, attr, var.attributes[attr])
        # Add coordinates attribute
        print(coordinates)
        setattr(varhandle, 'coordinates', ' '.join(coordinates[-2:]))
    f.close()


def vert_interp(var, heights):

    # Reinterpoles the hybrid vertical levels to altitude levels specified by the user
    cp = var.copy()
    if 'hyb_levels' not in var.coordinates.keys():
        print 'Reinterpolation to altitude levels works for variables which have hybrid layer coordinates'
        print 'No p-file_instances or z-file_instances, or horizontal 2D variables'
        return

    if 'z-levels' not in var.attributes.keys():
        print 'No z-levels attribute found, please assign the altitudes first using the class function assign_heights'
        return

    heights = np.asarray(heights, dtype = 'float32')
    data = np.asarray(var.data, dtype = 'float32')
    z_levels = np.asarray(var.attributes['z-levels'], dtype = 'float32')

    siz = data.shape


    if len(heights)==1:
        if var.dim == 2:
            interp_data=np.zeros((siz[1]))
            for j in range(0, siz[1]):
                vert_col = z_levels[:,j]
                interp_column = interp1_c.interp1(len(heights), vert_col,data[:,j],heights)[1][:]
                interp_column[heights<vert_col[-1]]=float('nan')
                interp_column[np.isinf(interp_column)]=float('nan')
                interp_data[j]=interp_column
        elif var.dim == 3:
            print 'Interpolating a 3-D variable vertically, this might take a while'
            print 'It is recommended to first slice your variable before you interpolate it...'
            interp_data=np.zeros((siz[1], siz[2]))
            for i in range(0, siz[1]):
                for j in range(0,siz[2]):
    #                f=interp.interp1d(z-levels[::-1,i,j],data[::-1,i,j], bounds_error=False, assume_sorted=True)
                    vert_col = z_levels[:,i,j]
                    interp_column=interp1_c.interp1(len(heights), vert_col,data[:,i,j],heights)[1][:]
                    interp_column[heights<vert_col[-1]]=float('nan')
                    interp_column[np.isinf(interp_column)]=float('nan')
                    interp_data[i,j]=interp_column
    else:
        if var.dim == 2:
            interp_data=np.zeros((len(heights),siz[1]))
            for j in range(0, siz[1]):
                vert_col = z_levels[:,j]
                interp_column=interp1_c.interp1(len(heights), vert_col,data[:,j],heights)[1][:]
                interp_column[heights<vert_col[-1]]=float('nan')
                interp_column[np.isinf(interp_column)]=float('nan')
                interp_data[:,j]=interp_column
        elif var.dim == 3:
            print 'Interpolating a 3-D variable vertically, this might take a while'
            print 'It is recommended to first slice your variable before you interpolate it...'
            if len(heights)==1:
                interp_data=np.zeros((siz[1], siz[2]))
            else:
                interp_data=np.zeros((len(heights),siz[1], siz[2]))
            for i in range(0, siz[1]):
                for j in range(0,siz[2]):
                    vert_col = z_levels[:,i,j]
                    interp_column=interp1_c.interp1(len(heights), vert_col,data[:,i,j],heights)[1][:]
                    interp_column[heights<vert_col[-1]]=float('nan')
                    interp_column[np.isinf(interp_column)]=float('nan')
                    interp_data[:,i,j]=interp_column

    if len(heights)==1:
        cp.dim=cp.dim-1
    else:
        cp.coordinates['heights']=heights
        # We need to replace the height key first
        cp.coordinates = move_element(cp.coordinates, "heights", 0)

    cp.data=interp_data
    cp.coordinates.pop('hyb_levels')
    cp.attributes.pop('z-levels')

    return cp

def WGS_to_COSMO(coords_WGS, SP_coords):
    if isinstance(coords_WGS, tuple):
        coords_WGS=np.vstack(coords_WGS)
    if isinstance(coords_WGS, np.ndarray ):
        if coords_WGS.shape[0]<coords_WGS.shape[1]:
            coords_WGS=coords_WGS.T
        lon = coords_WGS[:,1]
        lat = coords_WGS[:,0]
        input_is_array=True
    else:
        lon=coords_WGS[1]
        lat=coords_WGS[0]
        input_is_array=False

    SP_lon=SP_coords[1]
    SP_lat=SP_coords[0]


    lon = (lon*np.pi)/180 # Convert degrees to radians
    lat = (lat*np.pi)/180

    theta = 90+SP_lat # Rotation around y-axis
    phi = SP_lon # Rotation around z-axis

    phi = (phi*np.pi)/180 # Convert degrees to radians
    theta = (theta*np.pi)/180

    x = np.cos(lon)*np.cos(lat) # Convert from spherical to cartesian coordinates
    y = np.sin(lon)*np.cos(lat)
    z = np.sin(lat)

    x_new = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z
    y_new = -np.sin(phi)*x + np.cos(phi)*y
    z_new = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z

    lon_new = np.arctan2(y_new,x_new) # Convert cartesian back to spherical coordinates
    lat_new = np.arcsin(z_new)

    lon_new = (lon_new*180)/np.pi # Convert radians back to degrees
    lat_new = (lat_new*180)/np.pi

    if input_is_array:
        coords_COSMO = np.vstack((lat_new, lon_new)).T
    else:
        coords_COSMO=np.asarray([lat_new, lon_new])

    return coords_COSMO.astype('float32')



