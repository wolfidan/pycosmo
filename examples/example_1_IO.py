import pycosmo as pc


'''
Example 1: Input/Output
'''

#####################
# Opening a GRIB file
#####################


''' 
To open a file, you have to specify either grib (.grb, .grib) or 
netcdf (.nc, .cdf) files as input, if no suffix --> program will assume 
it is grib
'''

fname = './lfsf00132500'
file_h = pc.open_file(fname)

'''
The file handle has the same attributes as the NioFile type (https://www.pyngl.ucar.edu/Nio.shtml)
 + some others
'''

print(file_h) # Summary of the file
print(file_h.attributes) # Global attributes
print(file_h.variables.keys()) # Display all variables

#####################
# Loading variables
#####################


'''
You can use the get_variable function to extract a variable from the file,
and create a DataClass instance which will attach the right coordinates to it.
You need to specify a variable name, besides full names, wildcards are 
accepted as well as simplified keys specified in the operational grib key file
(c.f. grib_keys_cosmo.txt). In case of ambiguity the most likely variable will
be read.
'''

T = file_h.get_variable('T') # Read  temperature on hybrid levels
P = file_h.get_variable('P') # Read  pressure on hybrid levels

'''
You can also attach vertical levels (HHL) to the variables by using the
assign_heights function and by providing the path to the corresponding c (constant)
file. If no c-file is given, the function will look for one in the same folder
as the main file 
'''

T.assign_heights(cfile_name = './lfsf00000000c')
P.assign_heights(cfile_name = './lfsf00000000c')

'''
All of this could also be done in one line
'''

dic_vars = file_h.get_variable(['T','P'],assign_heights = True, cfile_name = './lfsf00000000c',
        shared_heights = False)

T = dic_vars['T']
P = dic_vars['P']


'''
When shared_heights == True, only the heights for the first variable will
be read and assigned to all variables of the list (this saves a bit of time)
'''

'''You can access the data of your variable by using either'''
data = T.data 
''' or '''
data = T[:]

'''
Simple operations can be performed directly on the DataClass instance
Simple operations include **,+,-,* and /. An advantage of this is that it will
not affect the data of the original variable(s)
'''
y = 2*T**2 + P

#####################
# Saving variables
#####################


'''
You can also save one or several variables to a NetCDF file (no GRIB support yet)
'''

pc.savevar([T,P],name = 'output_ex.nc')
