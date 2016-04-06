 # -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:31:55 2015

@author: wolfensb
"""
from scipy.io import netcdf
import matplotlib.pyplot as plt
import pyproj
from fnmatch import fnmatch
import file_class
import data_class
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import subprocess
import glob, os
import interp1_c


DERIVED_VARS=['PREC_RATE','QV_v','QR_v','QS_v','QG_v','QC_v','QI_v','QH_v',
              'QNR_v','QNS_v','QNG_v','QNC_v','QNI_v','QNH_v',
              'LWC','TWC','IWC','RHO','N','Pw','RELHUM']
             
                           

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
    
def check_if_variables_in_file(file_instance, varnames):
    for var in varnames:
        varname_checked=check_varname(file_instance,var)
        if varname_checked == '':
            return False
    return True
    
def check_varname(file_instance, varname):
    varname_checked=''    
    # First try to find variable in file_instance
    list_vars=np.asarray((file_instance.handle.variables.keys()))

    if varname in list_vars:
        varname_checked = varname
    else:  # Then try to find it using the grib key dictionary
        dic=get_grib_keys()
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
    
def format_ticks(labels,decimals=2):
    labels_f=labels
    for idx, val in enumerate(labels):
        if int(val) == val:  labels_f[idx]=val
        else: labels_f[idx]=round(val,decimals)
    return labels_f    


def get_constants():
    constants={}
    constants['cosmo_r_d']=287.05
    constants['cosmo_r_v']= 451.51
    constants['cosmo_rdv'] = constants['cosmo_r_d'] / constants['cosmo_r_v']
    constants['cosmo_o_m_rdv'] = 1.0 - constants['cosmo_rdv']
    constants['cosmo_rvd_m_o'] = constants['cosmo_r_v'] / constants['cosmo_r_d'] - 1.0
    return constants
    
    
def get_derived_var(file_instance, varname,get_proj_info):
    dic_csts=get_constants()
    derived_var=None
    if varname == 'PREC_RATE': # PRECIPITATION RATE
        try:
            d=get_variables(file,['PRR_GSP_GDS10_SFC','PRR_CON_GDS10_SFC','PRS_CON_GDS10_SFC','PRS_GSP_GDS10_SFC'],get_proj_info)
            derived_var=d['PRR_GSP_GDS10_SFC']+d['PRR_CON_GDS10_SFC']+d['PRS_CON_GDS10_SFC']+d['PRS_GSP_GDS10_SFC']
            if 'PRG_GSP_GDS10_SFC' in file.handle.variables.keys(): # Check if graupel is present
                derived_var+=get_variable(file,'PRG_GSP_GDS10_SFC',get_proj_info)
            if 'PRH_GSP_GDS10_SFC' in file.handle.variables.keys(): # Check if hail is present
                derived_var+=get_variable(file,'PRH_GSP_GDS10_SFC',get_proj_info)

            derived_var.name='PREC_RATE'
            derived_var.attributes['long_name']='precipitation intensity  [mm/s]'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'QV_v': # Water vapour mass density
        try:
            d=get_variables(file_instance,['QV','RHO'],get_proj_info)
            derived_var=d['QV']*d['RHO']
            derived_var.name='QV_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Water vapor mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'QR_v': # Rain water mass density
        try:
            d=get_variables(file_instance,['QR','RHO'],get_proj_info)
            derived_var=d['QR']*d['RHO']
            derived_var.name='QR_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Rain water mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'QS_v': # Snow water mass density
        try:
            d=get_variables(file_instance,['QS','RHO'],get_proj_info)
            derived_var=d['QS']*d['RHO']
            derived_var.name='QS_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Snow water mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'QG_v': # Graupel water mass density
        try:
            d=get_variables(file_instance,['QG','RHO'],get_proj_info)
            derived_var=d['QG']*d['RHO']
            derived_var.name='QG_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Graupel water mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'QC_v': # Cloud water mass density
        try:
            d=get_variables(file_instance,['QC','RHO'],get_proj_info)
            derived_var=d['QC']*d['RHO']
            derived_var.name='QC_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Cloud water mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'QI_v': # Ice cloud water mass density
        try:
            d=get_variables(file_instance,['QI','RHO'],get_proj_info)
            derived_var=d['QI']*d['RHO']
            derived_var.name='QI_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Ice cloud water mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'            
    elif varname == 'QH_v': # Hail water mass density
        try:
            d=get_variables(file_instance,['QH','RHO'],get_proj_info)
            derived_var=d['QH']*d['RHO']
            derived_var.name='QH_v'
            derived_var.attributes['units']='kg/m3'
            derived_var.attributes['long_name']='Hail water mass density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'   
    elif varname == 'QNR_v': # Rain number density
        try:
            d=get_variables(file_instance,['QNR','RHO'],get_proj_info)
            derived_var=d['QNR']*d['RHO']
            derived_var.name='QNR_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Rain number density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'   
    elif varname == 'QNS_v': # Snow number density
        try:
            d=get_variables(file_instance,['QNS','RHO'],get_proj_info)
            derived_var=d['QNS']*d['RHO']
            derived_var.name='QNS_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Snow number density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'   
    elif varname == 'QNG_v': # Graupel number density
        try:
            d=get_variables(file_instance,['QNG','RHO'],get_proj_info)
            derived_var=d['QNG']*d['RHO']
            derived_var.name='QNG_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Graupel number density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'   
    elif varname == 'QNC_v': # Cloud number density
        try:
            d=get_variables(file_instance,['QNC','RHO'],get_proj_info)
            derived_var=d['QNC']*d['RHO']
            derived_var.name='QNC_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Rain number density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'   
    elif varname == 'QNI_v': # Ice cloud particles number density
        try:
            d=get_variables(file_instance,['QNI','RHO'],get_proj_info)
            derived_var=d['QNI']*d['RHO']
            derived_var.name='QNI_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Ice cloud particles number density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'   
    elif varname == 'QNH_v': # Hail number density
        try:
            d=get_variables(file_instance,['QNH','RHO'],get_proj_info)
            derived_var=d['QNH']*d['RHO']
            derived_var.name='QNH_v'
            derived_var.attributes['units']='m^-3'
            derived_var.attributes['long_name']='Rain number density'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'               
    elif varname == 'LWC': # LIQUID WATER CONTENT
        try:
            d=get_variables(file_instance,['QC','QR'],get_proj_info)
            derived_var=d['QC']+d['QR']
            derived_var=derived_var*100000
            derived_var.name='LWC'
            derived_var.attributes['units']='mg/kg'
            derived_var.attributes['long_name']='Liquid water content'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'IWC': # ICE WATER CONTENT
        try:
            d=get_variables(file_instance,['QG','QS','QI'],get_proj_info)
            derived_var=d['QG']+d['QS']+d['QI']
            derived_var=derived_var*100000
            derived_var.name='IWC'
            derived_var.attributes['units']='mg/kg'
            derived_var.attributes['long_name']='Ice water content'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'
    elif varname == 'TWC': # TOTAL WATER CONTENT 
        try:
            d=get_variables(file_instance,['QG','QS','QI','QC','QV','QR'],get_proj_info)
            derived_var=d['QG']+d['QS']+d['QI']+d['QC']+d['QV']+d['QR']
            derived_var=derived_var*100000
            derived_var.name='TWC'
            derived_var.attributes['long_name']='Total water content'
            derived_var.attributes['units']='mg/kg'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'            
    elif varname == 'RHO': # AIR DENSITY
        try:
            d=get_variables(file_instance,['P','T','QV','QR','QC','QI','QS','QG'],get_proj_info)
            derived_var=d['P']/(d['T']*dic_csts['cosmo_r_d']*((d['QV']*dic_csts['cosmo_rvd_m_o']-d['QR']-d['QC']-d['QI']
            -d['QS']-d['QG'])+1.0))
            derived_var.name='RHO'
            derived_var.attributes['long_name']='Air density'
            derived_var.attributes['units']='kg/m3'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'       
    elif varname == 'Pw': # Vapor pressure
        try:
            d=get_variables(file_instance,['P','QV'],get_proj_info)
            derived_var=(d['P']*d['QV'])/(d['QV']*(1-0.6357)+0.6357)
            derived_var.attributes['long_name']='Vapor pressure'
            derived_var.attributes['units']='Pa'        
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'       
    elif varname == 'RELHUM': # Vapor pressure
        try:
            d=get_variables(file_instance,['Pw','T'],get_proj_info)
            esat=610.78*np.exp(17.2693882*(d['T'].data-273.16)/(d['T'].data-35.86)) # TODO
            derived_var=d['Pw']/esat*100
            derived_var.attributes['long_name']='Relative humidity'
            derived_var.attributes['units']='%'        
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'     
    elif varname == 'N': # Refractivity
        try:
            d=get_variables(file_instance,['T','Pw','P'],get_proj_info)
            derived_var=(77.6/d['T'])*(0.01*d['P']+4810*(0.01*d['Pw'])/d['T'])
            derived_var.attributes['long_name']='Refractivity'
            derived_var.attributes['units']='-'
        except:
            print 'Could not compute specified derived variable, check if all the necessary variables are in the input file_instance.'       
    else:
        print 'Could not compute derived variable, please specify a valid variable name:'
        print 'PREC_RATE, IWC, TWC, LWC or RHO'
    return derived_var 
    
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
    

def get_grib_keys():
    cur_path=os.path.dirname(os.path.realpath(__file__))
    f = open(cur_path+'/grib_keys.txt', 'r')
    dic={}
    for line in f:
        line=line.strip('\n')
        line=line.split(',')
        dic[line[0]]=line[1]
    return dic
    

def get_variable(file_instance, varname, get_proj_info=True):
    if varname in file_instance.dic_variables.keys():
        return file_instance.dic_variables[varname]
    else:
        print '--------------------------'
        print 'Reading variable '+varname
        if varname in DERIVED_VARS:
            var=get_derived_var(file_instance, varname,get_proj_info)
        else:
            varname_checked=check_varname(file_instance, varname)
            if varname_checked != '':
                var=data_class.Data_class(file_instance, varname_checked,get_proj_info)
                print 'Variable was read successfully'
            else:
                print 'Variable was not found in file_instance'
                var=None
        print '--------------------------' 
        print ''
        file_instance.dic_variables[varname]=var
    return var    

def get_variables(file_instance, list_varnames, get_proj_info=True, shared_heights=False, assign_heights=False, c_file=''):
    
    dic_var={}
    for i,v in enumerate(list_varnames):
        var=get_variable(file_instance, v, get_proj_info)
        if assign_heights:
            if i == 0 or not shared_heights:
                var.assign_heights(c_file)
            else: # If shared_heights is true we just copy the heights from the first variables to all others
                var.attributes['z-levels']=dic_var[list_varnames[0]].attributes['z-levels']
        dic_var[v]=var
    return dic_var

def hyb_avg(var):
    cp=var.copy()
    if not (var.file.type == 'h' or var.file.type == 'c'):
        print 'Averaging on hybrid layers only make sense for variables that are on hybrid levels'
        print 'No p-file_instances or z-file_instances, or horizontal 2D variables'
        return
    if var.dim == 3:
        cp.data=0.5*(var.data[0:-1,:,:] + var.data[1:,:,:])
    elif var.dim == 2:
        cp.data=0.5*(var.data[0:-2,:,:] + var.data[1:-1,:,:])
    cp.coordinates['hyb_levels']=var.coordinates['hyb_levels'][0:-2]
    return cp
    
 
def make_colorbar(fig,orientation='horizontal',label=''):

#    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    if orientation == 'horizontal':
        cbar_ax = fig['fig_handle'].add_axes([0.2, 0,0.6,1])
        axins = inset_axes(cbar_ax,
               width="100%", # width = 10% of parent_bbox width
               height="5%", # height : 50%
               loc=10,
               bbox_to_anchor=(0, -0.01, 1 , 0.15),
               bbox_transform=cbar_ax.transAxes,
               borderpad=0,
               )
    else:
        cbar_ax = fig['fig_handle'].add_axes([0, 0.2,1,0.6])
        axins = inset_axes(cbar_ax,
               width="5%", # width = 10% of parent_bbox width
               height="100%", # height : 50%
               loc=6,
               bbox_to_anchor=(1.01, 0, 0.15, 1),
               bbox_transform=cbar_ax.transAxes,
               borderpad=0,
               )
    cbar_ax.get_xaxis().tick_bottom()
    cbar_ax.axes.get_yaxis().set_visible(False)
    cbar_ax.axes.get_xaxis().set_visible(False)
    cbar_ax.set_frame_on(False)       
    
    cbar=plt.colorbar(cax=axins, orientation=orientation,label=label)

    levels = fig['cont_handle'].levels
    cbar.set_ticks(levels)
    cbar.set_ticklabels(format_ticks(levels,decimals=2))
    return cbar
    
    
def move_element(odict, thekey, newpos):
    odict[thekey] = odict.pop(thekey)
    i = 0
    for key, value in odict.items():
        if key != thekey and i >= newpos:
            odict[key] = odict.pop(key)
        i += 1
    return odict
    
def open_file(fname): # Just create a file_instance class
    return file_class.File_class(fname)
             

def overlay(list_vars, var_options=[{},{}], overlay_options={}):
    
    overlay_options_keys=overlay_options.keys()

    if 'labels' not in overlay_options_keys:
        overlay_options['labels']=[var.name for var in list_vars]
    if 'label_position' not in overlay_options_keys:
        overlay_options['label_position']='right'
   
    plt.hold(False)
    n_vars=len(list_vars)
    offsets=np.linspace(0.9,0.1,n_vars)
    basemap=''
    for idx, var in enumerate(list_vars):
        plt.hold(True)
        opt=var_options[idx]
        opt['no_colorbar'] = True 
        fig=var.plot(var_options[idx], basemap)
        basemap=fig['basemap']
        ax=plt.gca()
        box = ax.get_position()
        
        for pc in fig['cont_handle'].collections:
                proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_edgecolor()[0]) for pc in fig['cont_handle'].collections] 
        
        if overlay_options['label_position'] == 'right':
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd=plt.legend(proxy, format_ticks(opt['levels'],decimals=2),loc='center left', bbox_to_anchor=(1, offsets[idx]), title=overlay_options['labels'][idx])
        elif overlay_options['label_position'] == 'left':
            ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            lgd=plt.legend(proxy, format_ticks(opt['levels'],decimals=2),loc='center right', bbox_to_anchor=(-0.05, offsets[idx]), title=overlay_options['labels'][idx])
        elif overlay_options['label_position'] == 'top':
            ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
            lgd=plt.legend(proxy, format_ticks(opt['levels'],decimals=2),loc='lower center', bbox_to_anchor=(offsets[idx],1.05), title=overlay_options['labels'][idx])
        elif overlay_options['label_position'] == 'bottom':
            ax.set_position([box.x0, box.y0+0.05*box.height, box.width, box.height*0.8])
            lgd=plt.legend(proxy, format_ticks(opt['levels'],decimals=2),loc='upper center', bbox_to_anchor=(offsets[idx],-0.05), title=overlay_options['labels'][idx])                        
        ax.add_artist(lgd)
    return fig
       
def resize_domain(var, boundaries):
    # Boundaries format are [[lower_left_lat, lower_left_lon],[upper_right_lat, upper_right_lon]]
    if 'lat_2D' not in var.coordinates.keys() or 'lon_2D' not in var.coordinates.keys():
        print 'To resize the domain, the variable must be 2D or 3D and be georeferenced!'
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
        print 'Triming...command is:'
        print 'convert -density '+str(kwargs['dpi'])+' '+args[0]+' -trim +repage ' + args[0]
        subprocess.call('convert -density '+str(kwargs['dpi'])+' '+args[0]+' -trim +repage ' + args[0],shell=True)
    except:
        print 'Triming failed, check if imagemagick is installed'
        pass    
    return

def savevar(list_vars, name='output.nc'):
    try:
        f = netcdf.netcdf_file_instance(name, 'w')
    except:
        print 'Could not create or open file_instance '+name
        
    if isinstance(list_vars, data_class.Data_class):
        # Only one variable, put it into list
        list_vars=[list_vars]

    for var in list_vars:
        siz=var.data.shape
        print siz
        list_dim_names=[]
        for idx, dim in enumerate(var.coordinates.keys()):
            if dim + '_idx' not in f.dimensions.keys():
                f.createDimension(dim + '_idx', siz[idx])
            list_dim_names.append(dim + '_idx')
        varhandle=f.createVariable(var.name,'f', tuple(list_dim_names))
        varhandle[:]=var.data
        varhandle.coordinates=''
        for idx, dim in enumerate(var.coordinates.keys()):
            if dim not in f.variables.keys():
                if 'lon_2D' in dim:
                    if dim not in f.variables.keys():
                        dimhandle=f.createVariable(dim,'f', (dim.replace('lon_2D','lat_2D')+'_idx', dim+'_idx'))
                elif 'lat_2D' in dim:
                    if dim not in f.variables.keys():
                        dimhandle=f.createVariable(dim,'f', (dim+'_idx',dim.replace('lat_2D','lon_2D')+'_idx'))
                        
                else:
                    dimhandle=f.createVariable(dim,'f', (dim+'_idx',))
                    
                dimhandle[:]=var.coordinates[dim]
            
        for attr in var.attributes.keys():
            setattr(varhandle, attr, var.attributes[attr])
    f.close()
    
def vert_interp(var, heights):
    heights=np.asarray(heights).astype('float32')
    # Reinterpoles the hybrid vertical levels to altitude levels specified by the user
    cp=var.copy()
    if 'hyb_levels' not in var.coordinates.keys():
        print 'Reinterpolation to altitude levels works for variables which have hybrid layer coordinates'
        print 'No p-file_instances or z-file_instances, or horizontal 2D variables'
        return
    
    if 'z-levels' not in var.attributes.keys():
        print 'No z-levels attribute found, please assign the altitudes first using the class function assign_heights'
        return
        
    siz=var.data.shape

    if len(heights)==1:
        if var.dim == 2:
            interp_data=np.zeros((siz[1]))
            for j in range(0, siz[1]):
                vert_col=var.attributes['z-levels'][:,j]
                interp_column=interp1_c.interp1(len(heights), vert_col,var.data[:,j],heights)[1][:] 
                interp_column[heights<vert_col[-1]]=float('nan')
                interp_column[np.isinf(interp_column)]=float('nan')
                interp_data[j]=interp_column
        elif var.dim == 3:
            print 'Interpolating a 3-D variable vertically, this might take a while'
            print 'It is recommended to first slice your variable before you interpolate it...'
            interp_data=np.zeros((siz[1], siz[2]))
            for i in range(0, siz[1]):
                for j in range(0,siz[2]): 
    #                f=interp.interp1d(var.attributes['z-levels'][::-1,i,j],var.data[::-1,i,j], bounds_error=False, assume_sorted=True)
                    vert_col=var.attributes['z-levels'][:,i,j]
                    interp_column=interp1_c.interp1(len(heights), vert_col,var.data[:,i,j],heights)[1][:]    
                    interp_column[heights<vert_col[-1]]=float('nan')
                    interp_column[np.isinf(interp_column)]=float('nan')
                    interp_data[i,j]=interp_column
    else:
        if var.dim == 2:
            interp_data=np.zeros((len(heights),siz[1]))
            for j in range(0, siz[1]):
                vert_col=var.attributes['z-levels'][:,j]
                interp_column=interp1_c.interp1(len(heights), vert_col,var.data[:,j],heights)[1][:] 
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
    #                f=interp.interp1d(var.attributes['z-levels'][::-1,i,j],var.data[::-1,i,j], bounds_error=False, assume_sorted=True)
                    vert_col=var.attributes['z-levels'][:,i,j]
                    interp_column=interp1_c.interp1(len(heights), vert_col,var.data[:,i,j],heights)[1][:]    
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

