# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:59:47 2015

@author: wolfensb
"""
import pycosmo

import numpy as np
from collections import OrderedDict
import copy
from colormaps import get_colormap
import glob, os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import utilities
from matplotlib.colors import LogNorm
import datetime
import warnings

BASEMAPS={} # Dictionary of basemaps

class DataClass:
    # This is just a small class that contains the content of a variable, to facilitate manipulation of data.
    def __init__(self,file='', varname='',get_proj_info=True):
        if file != '' and varname != '':
            self.create(file,varname,get_proj_info)
            
    def create(self,file, varname, get_proj_info):
        self.file=file
        self.slice_type=None # Indicates if variable was sliced
        self.name=varname
        self.data=self.file.variables[varname][:].astype('float32')
        self.dim=len(self.data.shape) # Dimension of variable
        self.coordinates= OrderedDict() # Ordered dictionary because the order of coordinates is important
        self.attributes = self.file.variables[varname].__dict__
        self.dimensions = self.file.variables[varname].dimensions
        i=0
        siz=self.data.shape
        for dim in self.dimensions:
            try: # Check if corresponding variable exists
                if("_x_" in dim):
                    var_name_dim=dim.replace('_x_','_lat_') # latitude
                    res_i=self.file.variables[var_name_dim].__dict__['Di'] 
                    res_j=self.file.variables[var_name_dim].__dict__['Di'] 
                    # Add resolution to attributes                    
                    self.attributes['resolution']=[round(res_i[0]*1000)/1000,round(res_j[0]*1000)/1000] # Trick to bring resolutions to 2 digit precision
                    if get_proj_info:
                        dic_proj=self.file.variables[var_name_dim].__dict__
                    coord_name='lat_2D'
                elif("_y_" in dim):
                    var_name_dim = dim.replace('_y_','_lon_') # longitude
                    coord_name='lon_2D'
                elif("lv_HYBY" in dim): # hybrid level
                    var_name_dim=dim+'_l0' # level
                    coord_name='hyb_levels'
                elif("lv_HYBL" in dim): # hybrid level
                    var_name_dim=dim # level
                    coord_name='hyb_levels'
                elif('lv_ISBL0' in dim and self.file.type == 'p'): # pressure
                    var_name_dim=dim
                    coord_name='press_levels'
                elif('lv_GPML0' in dim and self.file.type == 'z'): # heights
                    var_name_dim=dim
                    coord_name='height_levels'                   
                self.coordinates[coord_name]=self.file.variables[var_name_dim][:].astype('float32')
            except: # Simply assign indexes going from 0 to the length of the dimension
#                print 'No data found for dimension, assigning indices...'
                self.coordinates[dim]=np.arange(0,siz[i]).astype('float32')
            i+=1
        
        init_time=datetime.datetime.strptime(self.attributes['initial_time'],'%m/%d/%Y (%H:%M)')
        step=int(self.attributes['forecast_time'][0])
        step_type=self.attributes['forecast_time_units']   
        if step_type == 'days':
            current_time=init_time+datetime.timedelta(days=step)
        elif step_type == 'hours':
            current_time=init_time+datetime.timedelta(hours=step)
        elif step_type == 'minutes':
            current_time=init_time+datetime.timedelta(minutes=step)
        elif step == 'seconds':
            current_time=init_time+datetime.timedelta(seconds=step)            
        
        self.attributes['time']=str(current_time)    
        if get_proj_info:
            self.attributes['proj_info']=dic_proj
        self.attributes.pop('coordinates',None) # This key is not needed anymore since we have already have the coordinates in the coordinates field
        try: # Compute domain
            self.attributes['domain_2D']=[[np.min(self.coordinates['lat_2D']), np.min(self.coordinates['lon_2D'])],[np.max(self.coordinates['lat_2D']),np.max(self.coordinates['lon_2D'])]]
        except:
            print 'Could not compute domain...'
            pass 
        
    def copy(self):
        cp=DataClass()
        cp.attributes=copy.deepcopy(self.attributes)
        cp.coordinates=copy.deepcopy(self.coordinates)
        for attr in self.__dict__.keys():
            if attr != 'file':
                setattr(cp,attr,copy.deepcopy(getattr(self,attr)))
            else: 
                setattr(cp,attr,getattr(self,attr))                
        return cp
        

    # Redefine operators
    
    def __getitem__(self,key):
        return self.data[key]
        
    def __add__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We sum by a scalar
            cp=self.copy()
            cp.data+=x
        elif isinstance(x, DataClass): # Sum by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data+=x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp

    def __radd__(self, x): # Reverse addition
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We sum by a scalar
            cp=self.copy()
            cp.data+=x
        elif isinstance(x, DataClass): # Sum by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data+=x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
    
    
    def __sub__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We subtract by a scalar
            cp=self.copy()
            cp.data-=x
        elif isinstance(x, DataClass): # Substract by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data-=x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp

    def __rsub__(self, x): # Reverse subtraction (non-commutative)
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We subtract by a scalar
            cp=self.copy()
            cp.data=x-cp.data
        elif isinstance(x, DataClass): # Substract by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data=x.data-cp.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp


    def __mul__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We muliply by a scalar
            cp=self.copy()
            cp.data*=x
        elif isinstance(x, DataClass): # Multiply by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data*=x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
        

    def __rmul__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We muliply by a scalar
            cp=self.copy()
            cp.data*=x
        elif isinstance(x, DataClass): # Multiply by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data*=x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
        
    def __div__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We divide by a scalar
            cp=self.copy()
            cp.data/=x
        elif isinstance(x, DataClass): # divide by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data/=x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
            
    def __rdiv__(self, x): # Reverse divsion (non-commutative)
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We divide by a scalar
            cp=self.copy()
            cp.data=x/cp.data
        elif isinstance(x, DataClass): # divide by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data=x.data/cp.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
            
    def __pow__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We divide by a scalar
            cp=self.copy()
            cp.data**x
        elif isinstance(x, DataClass): # divide by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data**x.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
    
    def __rpow__(self, x): # Reverse divsion (non-commutative)
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We divide by a scalar
            cp=self.copy()
            cp.data=x**cp.data
        elif isinstance(x, DataClass): # divide by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data=x.data**cp.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp        
            
    def __str__(self): # Redefines print in a nicely formatted style
        string='---------------------------------------------------\n'
        string+='Variable: '+self.name+', size='+str(self.data.shape)+', coords='+'('+','.join(self.coordinates.keys())+')\n'
        string+='Read from file: '+self.file.name+'\n'
        string+='Time : '+self.attributes['time']+'\n'
        string+='---------------------------------------------------\n'
        string+=str(self.data)+'\n'
        string+='---------------------------------------------------\n'
        string+='Coordinates:\n'
        for coord in self.coordinates.keys():
            string+='   '+coord+' : '+str(self.coordinates[coord].shape)+'\n'
            string+='   '+str(self.coordinates[coord])+'\n'
        string+='---------------------------------------------------\n'
        string+='Attributes:\n'
        for atr in self.attributes.keys():
            if atr == 'domain_2D': string+='   '+atr+' [[lat_bot_left, lon_bot_left], [lat_top_right, lon_top_right]] : "'+str(self.attributes[atr])+'"\n'
            else: string+='   '+atr+' : "'+str(self.attributes[atr])+'"\n'
        return string

        
    def plot(self, options={},basemap=''):

        fig={} # The output
        
        if self.dim == 3:
            print('No 3D plotting functions are implemented yet...sorry')
            print('Returning empty handle')
            return

        if 'cmap' not in options.keys():
            options['cmap']=get_colormap('jet')
        if 'scale' not in options.keys():
            options['scale']='linear'
        if 'filled' not in options.keys():
            options['filled'] = True
        if 'plot_altitudes' not in options.keys() or not self.slice_type in \
            ['lat','lon','latlon']:
            options['plot_altitudes'] = False
        if 'levels' not in options.keys():
            if options['filled']: num_levels=25
            else: num_levels=15
            if options['scale']=='log':
                options['levels']=np.logspace(np.nanmin(self[:].ravel()),
                                    np.nanmax(self[:].ravel()),num_levels)
            else:
                options['levels']=np.linspace(np.nanmin(self[:].ravel()), 
                                    np.nanmax(self[:].ravel()),num_levels)
        if 'no_colorbar' not in options.keys():
            options['no_colorbar']=False
        if 'cargs' not in options.keys():
            options['cargs']={}
        if 'cbar_title' not in options.keys():
            options['cbar_title']= self.attributes['units'] 

        
        coord_names=self.coordinates.keys()
        if 'lat_2D' in coord_names and 'lon_2D' in coord_names:
            spatial=True
        else: spatial=False

        mask = np.isnan(self.data)

        if spatial:
            m = self.get_basemap(basemap = basemap)
                
            x, y = m(self.coordinates['lon_2D'], self.coordinates['lat_2D']) # compute map proj coordinates
            
            m.contourf(x,y,mask,levels=[0.0,0.1,1],colors=['white','Grey'])
            if options['filled']:
                if options['scale'] == 'log':
                    vmin=min(options['levels'])
                    vmax=max(options['levels'])
                    data_no_zero=self.data
                    data_no_zero[data_no_zero<vmin]=0.00000001 # Hack to use log scale (won't be plotted)
                    CS = m.contourf(x,y,self.data, cmap=options['cmap'], 
                                  levels=options['levels'], vmin=vmin, vmax=vmax,
                                  norm=LogNorm(vmin=vmin, vmax=vmax),**options['cargs'])
                else:
                    CS=m.contourf(x,y,self.data, cmap=options['cmap'],levels=options['levels'],extend='max',  
                                  vmin=min(options['levels']),**options['cargs'])
            else:
                   
                mask = mask.astype(float)
                CS=m.contour(x,y,self.data, cmap=options['cmap'], 
                             levels=options['levels'],extend='max',
                             vmin=min(options['levels']),**options['cargs'])
                plt.clabel(CS, inline=1, fontsize=9)
                
            fig['basemap']=m
        else:
            if options['plot_altitudes']:
                try:
                    y = self.attributes['z-levels']
                    x = self.coordinates[coord_names[1]]
                    x = np.tile(x, (len(y),1))

                except:
                    print('Could not plot on altitude levels, plotting on model'+\
                          ' levels instead...')
                    options['plot_altitudes'] = False

                    
            if not options['plot_altitudes']:
                x=self.coordinates[coord_names[1]]
                y=self.coordinates[coord_names[0]]
            plt.contourf(x,y,mask, levels=[0.0,0.1,1],colors=['white','Grey'])     
            if options['filled']:
                if options['scale'] == 'log':
                     vmin=min(options['levels'])
                     vmax=max(options['levels'])
                     data_no_zero=self.data
                     data_no_zero[data_no_zero<vmin]=0.00000001 # Hack to use log scale (won't be plotted)
                     
                     CS=plt.contourf(x,y,self.data, cmap=options['cmap'],
                                     levels=options['levels'], vmin=vmin,
                                     vmax=vmax, norm=LogNorm(vmin=vmin, 
                                     vmax=vmax),**options['cargs'])
                else:
                     CS=plt.contourf(x,y,self.data, cmap=options['cmap'], levels=options['levels'], extend='max',vmin=min(options['levels']),**options['cargs'])
            else:
                 mask=mask.astype(float)
                 CS=plt.contour(x,y,self.data, cmap=options['cmap'], levels=options['levels'],extend='max', vmin=min(options['levels']),**options['cargs'])
                 plt.clabel(CS, inline=1, fontsize=9)
                 plt.xlabel(coord_names[1])
                 plt.ylabel(coord_names[0])

        plt.title(self.attributes['long_name'].capitalize()+' ['+self.attributes['units']+']')
       
        if not options['no_colorbar']:
            if options['filled']:
                cbar=plt.colorbar(fraction=0.046, pad=0.04, label=options['cbar_title'])
                cbar.set_ticks(options['levels'])
                cbar.set_ticklabels(utilities.format_ticks(options['levels'],decimals=2))
            else:
                ax=plt.gca()
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                for pc in CS.collections:
                    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_edgecolor()[0]) for pc in CS.collections] 
                lgd=plt.legend(proxy, utilities.format_ticks(options['levels'],decimals=2),loc='center left', bbox_to_anchor=(1, 0.6), title=options['cbar_title'])
                ax.add_artist(lgd)
        if options['plot_altitudes']:
            plt.ylabel('Altitude [m]')
            plt.gca().set_axis_bgcolor('Gray')
            
        fig['cont_handle']=CS
        fig['fig_handle']=plt.gcf()
        del options
       
        return fig


    def get_basemap(self, basemap = ''):
        domain=self.attributes['domain_2D']
        domain_str=str(domain)
        if domain_str in BASEMAPS.keys():
            basemap = BASEMAPS[domain_str]
        elif basemap == '':
            basemap = Basemap(projection='merc',
                              lon_0=0.5*(domain[0][1]+domain[1][1]),
                              lat_0=0.5*(domain[0][0]+domain[1][0]),\
            llcrnrlat=domain[0][0],urcrnrlat=domain[1][0],\
            llcrnrlon=domain[0][1],urcrnrlon=domain[1][1],\
            rsphere=6371200.,resolution='h',area_thresh=10000)
            BASEMAPS[domain_str] = basemap
             
        basemap.drawcoastlines()
        basemap.drawstates()
        basemap.drawcountries()
    
        # draw parallels.
        parallels = np.arange(int(0.8*domain[0][0]),int(1.2*domain[1][0]),1)
        basemap.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        # draw meridians
        meridians = np.arange(int(0.8*domain[0][1]),int(1.2*domain[1][1]),1)
        basemap.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
            
        return basemap
    
    def assign_heights(self, cfile_name=''):
        # The user can optionally specify a c-file that contains the constant parameters of the simulation (altitude, topography)
        # If no c-file is specified the function tries to find one in the folder of the file and if not possible use some standard c-files for Switzerland
        if cfile_name == '':
            print('No c-file specified, trying to find one in the same folder...')
            file_folder=os.path.dirname(self.file.name)
            cfile_names = glob.glob(file_folder+'*c.*')
            if(len(cfile_name)>1):
                cfile_name = cfile_names[0]
                cfile = pycosmo.open_file(cfile_name)
                print('c-file found in folder: '+cfile_name)
#            else:
#                print('No c-file found in folder, trying to use a standard-one over Switzerland')
#                
#                if 'resolution' in self.attributes.keys():
#                    if self.attributes['resolution'][0] == 0.02 and self.attributes['resolution'][1] == 0.02 : # COSMO-2
#                        cur_path=os.path.dirname(os.path.realpath(__file__))
#                        cfile_name = cur_path+'/constant_files/cosmo2_constant.nc'
#                        cfile = pycosmo.open_file(cfile_name)
#                    elif self.attributes['resolution'][0] == 0.07 and self.attributes['resolution'][1] == 0.07 : # COSMO-7
#                        cfile_name = cur_path+'/constant_files/cosmo7_constant.nc'
#                        cfile = pycosmo.open_file(cfile_name)    
#                    else:
#                        print('No c-file available for this resolution, aborting...')
#                        return
#                else:
#                    print('No resolution attribute found, cannot find corresponding reference c-file...')
#                    return
        else:
              cfile = pycosmo.open_file(cfile_name)
        
        # HHL is a 3D matrix that gives the height of every element
        HHL = cfile.get_variable('HH_GDS10_HYBL')

        # Check if heights are on the half levels, otherwise take the average
        if self.data.shape[0] < HHL.data.shape[0]: # ==> For full levels (W)
            HHL = hyb_avg(HHL)
        
            
        # Some variables (U,V) are defined on the edges of the grid boxes, and 
        # thus their coordinates do not correspond with the ones of HHL
        # DISCLAIMER: I am not too sure about this part...
        # For U
        if self.dimensions[1] == 'g10_x_3' and self.dimensions[2] == 'g10_y_4':
            HHL.data = 0.5*(HHL.data[:,:,0:-1] + HHL.data[:,:,1:])
            # Pad with NaN to account for the last column which is unknown
            HHL.data = np.pad(HHL.data,((0,0),(0,0),(0,1)),'constant',
                              constant_values=np.nan)
            # For V
        elif self.dimensions[1] == 'g10_x_5' and self.dimensions[2] == 'g10_y_6':
            HHL.data = 0.5*(HHL[:,0:-1,:] + HHL[:,1:,:])
            # Pad with NaN to account for the last row which is unknown
            HHL.data = np.pad(HHL.data,(0,),'constant')
            HHL.data = np.pad(HHL.data,((0,0),(0,1),(0,0)),'constant',
                              constant_values=np.nan)

        # Check if shapes agree
        if HHL.data.shape == self.data.shape:
            self.attributes['z-levels']=HHL.data.astype('float32')
            # Add info about associated c-file
            self.attributes['c-file'] = cfile_name
        else:
            warnings.warn('Heights found in the c-file do not correspond to size '\
            'of the data, could not assign heights!')
        return

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
    