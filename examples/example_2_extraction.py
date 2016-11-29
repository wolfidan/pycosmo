import pycosmo as pc
import matplotlib.pyplot as plt
import numpy as np

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
pc.plot(T_prof,options={'alt_coordinates':True}) # More info on plotting options of pycosmo in example 3


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
pc.plot(profiles[0],options={'alt_coordinates':True})
plt.subplot(2,1,2)
pc.plot(profiles[1],options={'alt_coordinates':True})

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
pc.plot(ppi_U)

plt.figure()

'''
Note that the coordinates of the PPI are 'range' and 'azimuth', however the lat, lon
distance at ground and height of every gate are also stored in the attributes
('lat_2D', 'lon_2D', 'altitude','dist_ground')
which makes it possible to plot the data in Cartesian coordinates
'''
y = ppi_U.attributes['lat_2D']
x = ppi_U.attributes['lon_2D']
plt.figure()
plt.contourf(x,y,ppi_U[:],levels=np.arange(0,15,1))
plt.xlabel('lat')
plt.ylabel('lon')

# OR even simpler with
plt.figure()
pc.plot(ppi_U,options={'alt_coordinates':True})

'''
RHI scans are supported as well, the inputs are basically the same as for PPI
except that azimuth is a mandatory scalar and elevations an optional vector.
The default for elevation is from 0 to 180 with a step corresponding to the
beamwidth
'''

options_RHI = {'beamwidth':1.5,'azimuth':5,
           'maxrange':15000,'rresolution':100,
           'rpos':[46.48,6.58,700],'npts_quad':[3,3],
           'refraction_method':1}
           
rhi_U = pc.extract(U,'RHI',options_RHI) 

plt.figure()
pc.plot(rhi_U,options={})


'''
Note that the coordinates of the RHI are 'range' and 'elevation', however the lat, lon
altitude and distance at ground of every gate are also stored in the attributes
('lat_2D', 'lon_2D', 'altitude','dist_ground')
which makes it possible to plot the data in Cartesian coordinates
'''
x = rhi_U.attributes['dist_ground']
y = rhi_U.attributes['altitude']

plt.figure()
plt.contourf(x,y,rhi_U[:],levels=np.arange(0,25,1),extend='both')
plt.xlabel('range')
plt.ylabel('altitude')

# OR even simpler with
plt.figure()
pc.plot(rhi_U,options={'alt_coordinates':True})

#####################
# Vertical interpolation
#####################

'''
For interpolation from hybrid levels to altitude levels, instead of extracting
slices by slices with the extract function, it is also possible to use the
vert_interp function to interpolate to several altitude levels at once
'''

alt_lvls = np.arange(500,5500,500) # Define 10 altitude levels from 500 to 5000
T_alt_levels = pc.vert_interp(T,alt_lvls)

print(T_alt_levels[:].shape) # shape is 10, 147, 147 as expected
print(T_alt_levels.coordinates['heights']) # The new coordinate "heights" replaces
# the hyb_levels coordinate

