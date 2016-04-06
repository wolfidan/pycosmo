
import numpy as np
import matplotlib.pyplot as plt
import pycosmo as pc
import os,time

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tictoc import *
plt.close('all')

    
if __name__=='__main__':
    print os.getcwd()
    # Open a file, you have to specify either grib (.grb, .grib) or netcdf (.nc, .cdf) files as input, if no suffix --> program will assume it is grib
    file_h=pc.open_file('./201403221955.grb')
#    print file_nc # Shows all variables contained in file
#    print file_nc.attributes # Global attributes
#    
    # Get cloud top
    tic()
    T=pc.get_variable(file_h,'T')
    T.assign_heights()
    toc()
#    P=pc.get_variable(file_h,'P')
#    
#    t0=time.time()
#    T=pc.vert_interp(T,np.arange(500,10000,200))
#    print t0-time.time()
#    index_PPI={'range':50000, 'radar_pos':[46,11,2000],'resolution':250, 'elevation':10}
    
#    T_slice=pc.slice(T,'PPI',index_PPI)
    
#    pc.savevar([T_slice, T],'test.nc')
    #QC=get_variable(file_nc,'QC')
    #savevar([QC, cloud], name='./testfds.nc') # Use this to save one or several variables to a netcdf (no GRIB support yet)

    # Resize the domain over Switzerland
#    plt.figure(figsize=(15,10))
#    options={}
    #options['levels']=np.arange(500,22000,1000)
#    options['cmap']=get_colormap('blues')
#    options['levels']=np.arange(10,200,5)
    #cloud_resized=resize_domain(cloud,[[45.7,5.7],[47.8,11]])  
#    IWC_slice.plot(options)
    
#    savefig('IWC.png',dpi=200) # Redefines the matplotlib savefig method, to trim the output with imagemagick if available
#
    #Load the first variable QR (specified rain content)
#    QR=get_variable(file_nc,'QR') # in kg/kg (so very small)
#    QR_mg=QR*100000 # Simple mathematical operators are supported (+-*/), the coordinates and attributes will be conserved
#    
##    QR_mg=get_variable(file_nc,'QR_mg') # Some variables can be specified as the user in the derived_variables function, this one for example is already implemenented (specific water contents in mg)
#    QR_mg.assign_heights(cfile='./lfff00000000c.nc') # This assigns heights to the model levels, we give the path of the COSMO c-file as argument (this file contains the level heights)
#    
#    # Similar for QG
#    QG=get_variable(file_nc,'QG')
#    QG_mg=QG*100000
#    QG_mg.assign_heights() # If no c-file is assigned, the function will first try to find a c-file in the same folder and if there is none, try to read a standard operational one from MeteoSwiss (might not be a good idea for you...;)     
#  
#    # Similar for QS
#    QS=get_variable(file_nc,'QS')
#    QS_mg=QS*100000
#    QS_mg.assign_heights() # If no c-file is assigned, the function will first try to find a c-file in the same folder and if there is none, try to read a standard operational one from MeteoSwiss (might not be a good idea for you...;)     
#    
#     # Similar for QC
#    QC=get_variable(file_nc,'QC')
#    QC_mg=QC*100000
#    QC_mg.assign_heights() # If no c-file is assigned, the function will first try to find a c-file in the same folder and if there is none, try to read a standard operational one from MeteoSwiss (might not be a good idea for you...;)     
#    
#    
#    # The library offers several options for slicing, at fixed latitude, longitude, along any coordinate profile, at a fixed height or at any model coordinate
#    slice_QG=slice(QG_mg,'level',2000) # Now we slice at a fixed altitude of 1000 m
#    slice_QR=slice(QR_mg,'level',2000)
#    slice_QC=slice(QC_mg,'level',2000)
#    slice_QS=slice(QS_mg,'level',2000) 
#    
#    # When you plot you can (optionally) specify some arguments in a python dictionary, you actually should, leaving default ones will be bad...
#    options={} # Now that you do not have to specify all (or even any) of these keys!
#    options['cmap']=get_colormap('greens') # You can specify a color map, the get_colormap function can help you
#    options['levels']=[10,25,50,100] # Define contour levels
#    options['filled']=False # Can be either true (default) or false
#    options['scale']='linear' # Can be linear or log (useful for precip...)
#    options['cargs']={"linewidths":1, 'linestyles':'-'} # Other arguments that must be given to the actual plotting method (contourf, contour) can be specified in a dictionary with a key called 'cargs'
#    options['no_colorbar']=False # Will be False by default
##    
#   # Plotting is easy!
#    # Now we create options for a second variable
#    options2=dict(options) # You have to use this to copy otherwise, the copy will be shallow
#    options2['cmap']=get_colormap([(247,244,249),(145,0,63)]) # You can also create a custom color map by giving RGB tuples (an arbitrary number), colors are then linearly or logaritmically interpolated between levels
#    options2['levels']=[5,10,15,25] # Define levels
#
#    # Now we create options for a second variable
#    options3=dict(options) # You have to use this to copy otherwise, the copy will be shallow
#    options3['cmap']=get_colormap('blues') # You can also create a custom color map by giving RGB tuples (an arbitrary number), colors are then linearly or logaritmically interpolated between levels
#    options3['levels']=[5,10,20,50] # Define levels
#
#    # Now we create options for a second variable
#    options4=dict(options) # You have to use this to copy otherwise, the copy will be shallow
#    options4['cmap']=get_colormap([(255,247,188),(140,45,4)]) # You can also create a custom color map by giving RGB tuples (an arbitrary number), colors are then linearly or logaritmically interpolated between levels
#    options4['levels']=[25,50,100,200]# Define levels
#    
#    
#    plt.figure(figsize=[20,20]) # Create plot handle as you would usually
#    overlay_options={}
#    overlay_options['labels']=['QR [mg/kg]]','QG [mg/kg]', 'QC [mg/kg]', 'QS [mg/kg]']
#    overlay_options['label_position']='top' # Can be top, bottom, right or left
#    overlay([slice_QR,slice_QG, slice_QC, slice_QS],[options, options2, options3, options4], overlay_options) # You can still change everything afterwards
#    plt.title('Mixing ratios at 2000 m altitude')
#    savefig('Q_overlay.png',dpi=200)
   
    # Extract temperature and slice at latitude = 46
    ################################################
#    T=get_variable(file_nc,'T')
#    T.assign_heights() # Assign heights because we want to further interpolate on height levels
#    slice_T_vert=slice(T,'lat',46) 
#
#    # Now reinterpolate to altitude levels
#    heights=np.arange(200,6000,100)
#    slice_T_vert_interp=vert_interp(slice_T_vert, heights)
#    plt.figure(figsize=[12,6])
#    options={}
#    options['levels']=np.arange(250,280,2)
#    options['cmap']=get_colormap('temp')
#    slice_T_vert_interp.plot(options) # Again we could use options as before
#    # Note that if anything is not ok in the plot you can still change it afterwards!
#    plt.title('Temperature [K] along Lat=46 deg')
#    plt.xlabel('Longitude')
#    plt.ylabel('Altitude [m]')
#    savefig('T_46.png',bbox_inches='tight',dpi=200)
#    
#    
#    # Extract pressure and slice at user-specified profile
#    ################################################
#    QS=get_variable(file_nc,'QS')
#    QS.assign_heights() # Assign heights because we want to further interpolate on height levels
#    QS=QS*100000
#    profile=mes_profile([44,7], [46,9], step=1000) # We get a profile of lat/lon coordinates from 45N/7E to 46N/8E with a step of 1000 meters
#    profile2=coords_profile([47,8.28], [45,9.16], step=2000) # We get a profile of 500 points along a profile given by lat/lon coordinates from 45N/7E to 46N/8E 
#
#    slice_P_vert_2=slice(QS,'latlon',profile2)
#    plt.plot(slice_P_vert_2.attributes['z-levels'][-1,:])
#    plt.show()
#    # Now reinterpolate to altitude levels
#    heights=np.arange(0,10000,50)
#    slice_P_vert_interp2=vert_interp(slice_P_vert_2, heights)
#    
#    options={}
#    options['cmap']=get_colormap('reds') # Will be jet by default
#    options['levels']=np.arange(10,70,10)
#    options['filled']=False
#    options['cbar_title']='QS [mg/kg]'
#    
#    plt.figure(figsize=[12,6])
#    
#    slice_P_vert_interp2.plot(options) # Again we could use options as before
#    plt.grid()
#    plt.title('QS profile from Luzern to Milano')
#    plt.xlabel('Distance [m]')
#    plt.ylabel('Altitude [m]')
#    savefig('QS_profile.png',dpi=200)
###    
#    # Panel plots of precipitation intensities on log scale
#    ################################################
##
#    precip=get_variable(file_nc,'TOT_PREC*')
#
#    f=plt.figure(figsize=(20,13))
#    options={}
#    options['cmap']=get_colormap('precip') # Will be jet by default
#    options['levels']=[0.05,0.1,0.2,0.5,1,2,5,10,20,50]
#    options['scale']='log' # Can be 'log' or 'linear' (default)
#    options['filled']=True # Can be either true (default) or false
#    options['no_colorbar']=True
#    plt.subplot(2,2,1)
#    fig=precip.plot(options)
#    plt.title('Case 1')
#    plt.subplot(2,2,2)
#    fig=precip.plot(options)
#    plt.title('Case 2')
#    plt.subplot(2,2,3)
#    fig=precip.plot(options)
#    plt.title('Case 3')
#    plt.subplot(2,2,4)
#    fig=precip.plot(options)    
#    plt.title('Case 4')
#    cbar=make_colorbar(fig, orientation='horizontal', label='[mm]')
#    plt.suptitle('Panels of precipitation amounts') # Overall title
#
#    savefig('precip_panel.png',dpi=200)
