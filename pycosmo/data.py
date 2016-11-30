# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:59:47 2015

@author: wolfensb
"""
import pycosmo

import numpy as np
from collections import OrderedDict
import copy
import glob, os
import datetime
import warnings

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
            self.attributes['z-levels'] = HHL.data.astype('float32')
            # Add info about associated c-file
            self.attributes['c-file'] = cfile_name
        else:
            warnings.warn('Heights found in the c-file do not correspond to size '\
            'of the data, could not assign heights!')
        
        cfile.close()
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
    