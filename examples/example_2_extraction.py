import pycosmo as pc
import matplotlib.pyplot as plt

'''
Example 2: Extracting slices
'''

fname = './lfsf00132500'
file_h = pc.open_file(fname)

T = file_h.get_variable('T') # Read  temperature on hybrid levels
P = file_h.get_variable('P') # Read  pressure on hybrid levels
U = file_h.get_variable('U') # Read  U-comp of wind on hybrid levels

T.assign_heights(cfile_name = './lfsf00000000c')
P.assign_heights(cfile_name = './lfsf00000000c')
U.assign_heights(cfile_name = './lfsf00000000c')

#####################
# Extracting slices
#####################


'''
The extract function offers many way to extract slices of data from COSMO
variables

The following types of extractions are possible:
    1) 'i' :  Cut at fixed zero-based i-index (no interpolation)
    2) 'j' :  Cut at fixed zero-based j-index (no interpolation)
    3) 'k' :  Cut at fixed zero-based k-index (no interpolation)
    4) 'level': Cut at fixed level (altitude or pressure) (with interpolation)
    5)  'lat' : Cuts a profile at fixed latitude (with interpolation)
    6) 'lon' : Cuts a profile at fixed longitude (with interpolation)
    7) 'latlon': Cuts a profile between two arbitrary coordinates (with interpolation)
    8) 'PPI' : Simulates a PPI scan from a radar with atm. refr and beam broadening.
    9) 'RHI' : Simulates a RHI scan from a radar with atm. refr and beam broadening.
'''


#####################
# Vertical profiles
#####################

'''
We start with a profile of temperature at fixed latitude
'''

T_prof = pc.extract(T,'lat',47)

'''
Plot the slice
'''
T_prof.plot(options={'plot_altitudes':True}) # More info on plotting options of pycosmo in example 3


'''
Now we do a profile from Payerne to Sion, with one point every 1000 m,
 we first need to create a coordinate profile with the coords_profile function
'''

coords_payerne = [46.81321, 6.94272] # lat/lon
coords_sion = [46.22007, 7.33247] # lat/lon

coords_prof = pc.coords_profile(coords_payerne,coords_sion,step=1000)

'''
Then we use this coords profile in the extract function
'''

profiles = pc.extract([T,P],'latlon',coords_prof) # You can extract several variables at once!

plt.figure()
plt.subplot(2,1,1)
profiles[0].plot(options={'plot_altitudes':True})
plt.subplot(2,1,2)
profiles[1].plot(options={'plot_altitudes':True})

#####################
# Radar simulation
#####################

'''
The PPI option of extract allows to simulate a radar PPI, the input is a dictionnary
with the following keys

rpos : the coordinates of the radar [lat,lon,alt] 

rresolution : the radial resolution of the radar

beamwidth : the 3dB beamwidth of the radar antenna

maxrange : the maximal range of the radar

elevation : the elevation angle (theta) from the ground

azimuth (optional) : the vector of azimuth angles (one for every radial), 
                     default is from 0 to 360 with a step corresponding
                     to the beamwidth
                     
npts_quad (optional) : tuple (n,m) with the number of sub-beams to consider
                       in every direction (n : horizontal, m : vertical) 
                       when integrating over the antenna
                       power density, the larger the more accurate the
                       integration. when [1,1] is chosen only the central beam 
                       is considerd (no integration)

refraction_method (optional) : the method used to compute the atmospheric 
                               refraction, if 1 : the simple 4/3 Earth radius
                               model is used.
                               If 2: the method by Zeheng and Blahak (2014) is 
                               used, this requires, pressure, temperature and
                               relative humidity to be present in the COSMO
                               file
'''

options_PPI = {'beamwidth':1.5,'elevation':5,
           'maxrange':15000,'rresolution':100,
           'rpos':[46.48,6.58,700],'npts_quad':[3,3],
           'refraction_method':1}

ppi_U = pc.extract(U,'PPI',options_PPI) 

plt.figure()
ppi_U.plot()



'''
RHI scans are supported as well, the inputs are basically the same as for PPI
except that azimuth is a mandatory scalar and elevations an optional vector.
The default for elevation is from 0 to 180 with a step corresponding to the
beamwidth
'''

options_RHI = {'beamwidth':1.5,'azimuth':5,
           'maxrange':15000,'rresolution':100,
           'rpos':[46.48,6.58,700],'npts_quad':[1,1],
           'refraction_method':1}
           
rhi_U = pc.extract(T,'RHI',options_RHI) 

plt.figure()
rhi_U.plot(options={})




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
