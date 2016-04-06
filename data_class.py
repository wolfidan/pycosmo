# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:59:47 2015

@author: wolfensb
"""
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
from file_class import File_class

BASEMAPS={} # Dictionary of basemaps

class Data_class:
    # This is just a small class that contains the content of a variable, to facilitate manipulation of data.
    def __init__(self,file='', varname='',get_proj_info=True):
        if file != '' and varname != '':
            self.create(file,varname,get_proj_info)
            
    def create(self,file, varname, get_proj_info):
        self.file=file
        self.slice_type=None # Indicates if variable was sliced
        self.name=varname
        self.data=self.file.handle.variables[varname][:].astype('float32')
        self.dim=len(self.data.shape) # Dimension of variable
        self.coordinates= OrderedDict() # Ordered dictionary because the order of coordinates is important
        self.attributes=self.file.handle.variables[varname].__dict__
        dimensions=self.file.handle.variables[varname].dimensions
        i=0
        siz=self.data.shape
        for dim in dimensions:
            try: # Check if corresponding variable exists
                if("_x_" in dim):
                    var_name_dim=dim.replace('_x_','_lat_') # latitude
                    res_i=self.file.handle.variables[var_name_dim].__dict__['Di'] # Trick to bring resolutions to 2 digit precision
                    res_j=self.file.handle.variables[var_name_dim].__dict__['Di'] # This is to avoid floating point issues during comparisons
                    # Add resolution to attributes                    
                    self.attributes['resolution']=[round(res_i[0]*1000)/1000,round(res_j[0]*1000)/1000]
                    if get_proj_info:
                        dic_proj=self.file.handle.variables[var_name_dim].__dict__
                    coord_name='lat_2D'
                elif("_y_" in dim):
                    var_name_dim=dim.replace('_y_','_lon_') # longitude
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
                self.coordinates[coord_name]=self.file.handle.variables[var_name_dim][:].astype('float32')
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
        cp=Data_class()
        cp.attributes=copy.deepcopy(self.attributes)
        cp.coordinates=copy.deepcopy(self.coordinates)
        for attr in self.__dict__.keys():
            if attr != 'file':
                setattr(cp,attr,copy.deepcopy(getattr(self,attr)))
            else: 
                setattr(cp,attr,getattr(self,attr))                
        return cp
            
    # Redefine operators
    def __add__(self, x):
        cp=None
        if isinstance(x,(int,float,bool,np.ndarray )): # We sum by a scalar
            cp=self.copy()
            cp.data+=x
        elif isinstance(x, Data_class): # Sum by another variable
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
        elif isinstance(x, Data_class): # Sum by another variable
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
        elif isinstance(x, Data_class): # Substract by another variable
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
        elif isinstance(x, Data_class): # Substract by another variable
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
        elif isinstance(x, Data_class): # Multiply by another variable
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
        elif isinstance(x, Data_class): # Multiply by another variable
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
        elif isinstance(x, Data_class): # divide by another variable
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
        elif isinstance(x, Data_class): # divide by another variable
            if self.data.shape == x.data.shape and self.coordinates.keys() == x.coordinates.keys():
                cp=self.copy()
                cp.data=x.data/cp.data
                keys=self.attributes.keys()
                for att in x.attributes.keys():
                    if att not in keys:
                        cp.attributes[att]=x.attributes[att]
        return cp
            
            
    def __str__(self): # Redefines print in a niceted style
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
            print 'No 3D plotting functions are implemented yet...sorry'
            print 'Returning empty handle'
            return

        options_keys=options.keys()

        if 'cmap' not in options_keys:
            options['cmap']=get_colormap('jet')
        if 'scale' not in options_keys:
            options['scale']='linear'
        if 'filled' not in options_keys:
            options['filled']=True
        if 'levels' not in options_keys:
            if options['filled']: num_levels=25
            else: num_levels=15
            if options['scale']=='log':
                options['levels']=np.logspace(np.nanmin(self.data.ravel()), np.nanmax(self.data.ravel()),num_levels)
            else:
                options['levels']=np.linspace(np.nanmin(self.data.ravel()), np.nanmax(self.data.ravel()),num_levels)
                
        if 'no_colorbar' not in options_keys:
            options['no_colorbar']=False
        if 'cargs' not in options_keys:
            options['cargs']={}
        if 'cbar_title' not in options_keys:
            options['cbar_title']=''

            
        coord_names=self.coordinates.keys()
        if 'lat_2D' in coord_names and 'lon_2D' in coord_names:
            spatial=True
        else: spatial=False

        mask=np.isnan(self.data)

        if spatial:
           if basemap == '':
               m=self.get_basemap()
           else:
               m=basemap
               
           x, y = m(self.coordinates['lon_2D'], self.coordinates['lat_2D']) # compute map proj coordinates
           if options['filled']:
               if options['scale'] == 'log':
                   vmin=min(options['levels'])
                   vmax=max(options['levels'])
                   data_no_zero=self.data
                   data_no_zero[data_no_zero<vmin]=0.00000001 # Hack to use log scale (won't be plotted)
                   m.contourf(x,y,mask, levels=[0.1,1.1],colors=['LightGray','#000000'])
                   CS=m.contourf(x,y,self.data, cmap=options['cmap'], levels=options['levels'], vmin=vmin, vmax=vmax, norm=LogNorm(vmin=vmin, vmax=vmax),**options['cargs'])
               else:
                   m.contourf(x,y,mask, levels=[0.1,1.1],colors=['LightGray','#000000'])
                   CS=m.contourf(x,y,self.data, cmap=options['cmap'], levels=options['levels'],extend='max', vmin=min(options['levels']),**options['cargs'])
           else:
               mask=mask.astype(float)
               m.contourf(x,y,mask, levels=[0.1,1.1],colors=['LightGray','#000000'])
               CS=m.contour(x,y,self.data, cmap=options['cmap'], levels=options['levels'],extend='max', vmin=min(options['levels']),**options['cargs'])
               plt.clabel(CS, inline=1, fontsize=9)
           fig['basemap']=m
        else:
            x=self.coordinates[coord_names[1]]
            y=self.coordinates[coord_names[0]]
            if options['filled']:
                
               if options['scale'] == 'log':
                   vmin=min(options['levels'])
                   vmax=max(options['levels'])
                   data_no_zero=self.data
                   data_no_zero[data_no_zero<vmin]=0.00000001 # Hack to use log scale (won't be plotted)
                   plt.contourf(x,y,mask, levels=[0.1,1.1],colors=['LightGray','#000000'])
                   CS=plt.contourf(x,y,self.data, cmap=options['cmap'], levels=options['levels'], vmin=vmin, vmax=vmax, norm=LogNorm(vmin=vmin, vmax=vmax),**options['cargs'])
               else:
                   plt.contourf(x,y,mask, levels=[0.1,1.1],colors=['LightGray','#000000'])
                   CS=plt.contourf(x,y,self.data, cmap=options['cmap'], levels=options['levels'], extend='max',vmin=min(options['levels']),**options['cargs'])
            else:
               mask=mask.astype(float)
               plt.contourf(x,y,mask, levels=[0.1,1.1],colors=['LightGray','#000000'])
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
 
        fig['cont_handle']=CS
        fig['fig_handle']=plt.gcf()
        return fig


    def get_basemap(self):
        domain=self.attributes['domain_2D']
        domain_str=str(domain)
        if domain_str in BASEMAPS.keys():
            m=BASEMAPS[domain_str]
        else:
            m = Basemap(projection='merc',lon_0=0.5*(domain[0][1]+domain[1][1]),lat_0=0.5*(domain[0][0]+domain[1][0]),\
            llcrnrlat=domain[0][0],urcrnrlat=domain[1][0],\
            llcrnrlon=domain[0][1],urcrnrlon=domain[1][1],\
            rsphere=6371200.,resolution='h',area_thresh=10000)
            BASEMAPS[domain_str]=m
            
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()
    
        # draw parallels.
        parallels = np.arange(int(0.8*domain[0][0]),int(1.2*domain[1][0]),1)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        # draw meridians
        meridians = np.arange(int(0.8*domain[0][1]),int(1.2*domain[1][1]),1)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
            
        return m
    
    def assign_heights(self, cfile=''):
        # The user can optionally specify a c-file that contains the constant parameters of the simulation (altitude, topography)
        # If no c-file is specified the function tries to find one in the folder of the file and if not possible use some standard c-files for Switzerland
        if cfile == '':
            print 'No c-file specified, trying to find one in the same folder...'
            file_folder=os.path.dirname(self.file.name)
            c_files=glob.glob(file_folder+'*c.*')
            if(len(c_files)>1):
                cfile=utilities.open_file(c_files[0])
                print 'c-file found in folder: '+c_files[0]
            else:
                print 'No c-file found in folder, trying to use a standard-one over Switzerland'
                
                if 'resolution' in self.attributes.keys():
                    if self.attributes['resolution'][0] == 0.02 and self.attributes['resolution'][1] == 0.02 : # COSMO-2
                        cur_path=os.path.dirname(os.path.realpath(__file__))
                        cfile=File_class(cur_path+'/constant_files/cosmo2_constant.nc')
                    elif self.attributes['resolution'][0] == 0.07 and self.attributes['resolution'][1] == 0.07 : # COSMO-7
                        cfile=File_class(cur_path+'/constant_files/cosmo7_constant.nc')
                    else:
                        print 'No c-file available for this resolution, aborting...'
                        return
                else:
                    print 'No resolution attribute found, cannot find corresponding reference c-file...'
                    return
        else:
              cfile=utilities.open_file(cfile)
        
        # HHL is a 3D matrix that gives the height of every element
        HHL = utilities.get_variable(cfile, 'HH_GDS10_HYBL')

        # Check if heights are one the same hybrid levels, otherwise take the center
        #print HHL.data.shape[0], self.data.shape[0]
        if self.data.shape[0] < HHL.data.shape[0]: # ==> More levels in the vertical heights
            
            HHL=utilities.hyb_avg(HHL)

        self.attributes['z-levels']=HHL.data.astype('float32')

        return

