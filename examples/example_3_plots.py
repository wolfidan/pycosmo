import pycosmo as pc
import matplotlib.pyplot as plt
import numpy as np

'''
Example 3: Plotting data
'''

fname = './lfsf00132500'
cname = './lfsf00000000c'
file_h = pc.open_file(fname)

T = file_h.get_variable('T') # Read  temperature on hybrid levels
P = file_h.get_variable('P') # Read  pressure on hybrid levels
U = file_h.get_variable('U') # Read  U-comp of wind on hybrid levels

T.assign_heights(cfile_name = cname)
P.assign_heights(cfile_name = cname)
U.assign_heights(cfile_name = cname)

#####################
# Plotting spatial 2D data
#####################


'''
First we extract a slice of temperatures at 3000 m
'''
T_3000 = pc.extract(T,'level',3000)
U_3000 = pc.extract(U,'level',3000)

'''
When you plot you can (optionally) specify some arguments in a python dictionary, 
you actually should, leaving default ones will be bad...
Below is the list of options keys and their defaults
options={} # Note that you do not have to specify all (or even any) of these keys!
options['cmap']=get_colormap('greens') # You can specify a color map, the get_colormap function can help you
options['levels']=[10,25,50,100] # Define contour levels
options['filled']=False # Can be either true (default) or false
options['scale']='linear' # Can be linear or log (useful for precip...)
options['cargs']={"linewidths":1, 'linestyles':'-'} # Other arguments
that must be given to the actual plotting method (contourf, contour)
can be specified in a dictionary with a key called 'cargs'
options['no_colorbar'] = False # IF you want to colorbar
options['alt_coordinates'] = False # When possible will use an alternative
coordinate system for plotting: for profiles (lat, lon, latlon), this will plot
altitudes instead of hybrid levels in the vertical, for PPI scans  this will
plot the data in cartesian coordinates (lat/lon), for RHI scans this will plot
the data in cartesian coordinates (distance and altitude)
'''

plt.figure()
options={}
options['levels'] = np.arange(500,22000,1000)
options['cmap'] = pc.get_colormap('RdBu_r')
'''
The get_colormap function (in the colormaps.py module) allows either
to load matplotlib colormaps or to create custom ones by specifying a list of
colors
'''
options['levels'] = np.arange(260,270,1)
plot = T_3000.plot(options) # Areas where topo > level are shown in grey
    
'''
Note that you can still change things on the plot after calling .plot()
Note also that when lat_2D and lon_2D are coordinates of the data, the data
will be plotted on a basemap which takes some time to generate. Everytime a 
basemap is generated it is stored in memory and so further plots defined on the
same domain are plotted faster... You can also explicitely retrieve a basemap
from the handle returned by the DataClass.plot() function and reuse as input for
a new plot
'''
#plt.figure()
#U_3000.plot(basemap=plot['basemap']) # Areas where topo > level are shown in grey

#####################
# Custom colormaps
#####################


QV = file_h.get_variable('QV',assign_heights = True, cfile_name = cname) * 1000
QV.attributes['units'] = 'g/kg'
QV_prof = pc.extract(QV,'lat',47)

# Now we create options for a second variable
options2 = dict(options) # You have to use this to copy an existing dic, otherwise, the copy will be shallow
'''
# You can also create a custom color map by giving RGB tuples (an arbitrary number), 
colors are then linearly or logaritmically interpolated between levels
'''

# Use log = True, to create a colorscale with logarithmic interpolation
# between colors

options2['cmap']= pc.get_colormap([(216,179,101),(245,245,245),(90,180,172)],log=False) 
options2['levels']= np.arange(0,5,0.2) # Define levels
options2['plot_altitudes']=True # Define levels

plt.figure()
QV_prof.plot(options2) #  Areas where topo > level are shown in light grey


#####################
# Panel plots of precip
#####################

precip = file_h.get_variable('TOT_PREC*')

f=plt.figure(figsize=(20,13))
options={}
options['cmap'] = pc.get_colormap('precip') # Will be jet by default
options['levels']=[0.05,0.1,0.2,0.5,1,2,5,10,20,50]
options['scale']='log' # Can be 'log' or 'linear' (default)
options['filled']=True # Can be either true (default) or false
options['no_colorbar']=True
plt.subplot(2,2,1)
fig=precip.plot(options)
plt.title('Case 1')
plt.subplot(2,2,2)
fig=precip.plot(options)
plt.title('Case 2')
plt.subplot(2,2,3)
fig=precip.plot(options)
plt.title('Case 3')
plt.subplot(2,2,4)
fig=precip.plot(options)    
plt.title('Case 4')
cbar= pc.make_colorbar(fig, orientation='horizontal', label='[mm]')
plt.suptitle('Panels of precipitation accumulations [mm]') # Overall title

'''
The savefig function of pycosmo overrides pyplot's savefig, by cropping
the plot as the end (if imagemagick is installed)
'''
pc.savefig('precip_panel.png',dpi=200)


#####################
# Overlay of plots
#####################

#Load the first variable QR (specified rain content)

QR = file_h.get_variable('QR') # in kg/kg (so very small)
QR_mg=QR*100000 # Simple mathematical operators are supported (+-*/), the coordinates and attributes will be conserved

QR_mg.assign_heights(cfile_name=cname)
    
# Similar for QG
QG = file_h.get_variable('QG')
QG_mg = QG*100000
QG_mg.assign_heights(cfile_name = cname) 
  
# Similar for QS
QS = file_h.get_variable('QS')
QS_mg=QS*100000
QS_mg.assign_heights(cfile_name = cname) 

# Similar for QC
QC = file_h.get_variable('QC')
QC_mg=QC*100000
QC_mg.assign_heights(cfile_name = cname) 


    
slice_QG = pc.extract(QG_mg,'lon',8) # Now we slice at a fixed lon of 8
slice_QR = pc.extract(QR_mg,'lon',8)
slice_QC = pc.extract(QC_mg,'lon',8)
slice_QS = pc.extract(QS_mg,'lon',8) 
    
options={} # Now that you do not have to specify all (or even any) of these keys!
options['cmap']=pc.get_colormap('Greens') # You can specify a color map, the get_colormap function can help you
options['levels']=[0.2,0.5,1,2,5] # Define contour levels
options['filled']=False # Can be either true (default) or false
options['scale']='linear' # Can be linear or log (useful for precip...)
options['cargs']={"linewidths":1, 'linestyles':'-'} # Other arguments that must be given to the actual plotting method (contourf, contour) can be specified in a dictionary with a key called 'cargs'
options['no_colorbar']=False # Will be False by default
options['plot_altitudes'] = True

# Now we create options for a second variable
options2=dict(options) 
options2['cmap'] = pc.get_colormap('Reds')  # You can also create a custom color map by giving RGB tuples (an arbitrary number), colors are then linearly or logaritmically interpolated between levels
options2['levels']=[0.5,1,2,5,10] # Define levels

# Now we create options for a second variable
options3=dict(options) 
options3['cmap'] = pc.get_colormap('Blues') # You can also create a custom color map by giving RGB tuples (an arbitrary number), colors are then linearly or logaritmically interpolated between levels
options3['levels']=[0.5,1,2,5,10] # Define levels

# Now we create options for a fourth variable
options4=dict(options)
options4['cmap'] = pc.get_colormap('Greys')  # You can also create a custom color map by giving RGB tuples (an arbitrary number), colors are then linearly or logaritmically interpolated between levels
options4['levels']=[10,25,50,75,100]# Define levels
    
    
plt.figure(figsize=[20,20]) # Create plot handle as you would usually
overlay_options={}
overlay_options['labels']=['QR [mg/kg]]','QG [mg/kg]', 'QC [mg/kg]', 'QS [mg/kg]']
overlay_options['label_position']='top' # Can be top, bottom, right or left
pc.overlay([slice_QR,slice_QG, slice_QC, slice_QS],[options, options2, options3, options4], overlay_options) # You can still change everything afterwards
plt.title('Specific water contents at 8 deg. lon.')
pc.savefig('Q_overlay.png',dpi=200)
