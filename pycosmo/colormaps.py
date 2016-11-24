# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:30 2015

@author: wolfensb
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''
get_colormap(col_type, position = None, log = False)

Creates a colormap instance to be used in the DataClass.plot() function, either
by getting a matplotlib colormap or by creating a new one depending on the input

Inputs :
col_type -> can be either a string or a list of 3D tuples (one for every color)
            string has to be one of matplotlib's colormaps (http://matplotlib.org/examples/color/colormaps_reference.html)

position ->  The position of every color along the color scale
log      -> log: if True a log scale is used to define the positions of the colors
'''

def get_colormap(col_type, position = None, log = False):
    try:
        if col_type == 'precip':
            colors = [(43,66,181), (67,222,139), (245,245,45), (252,45,45)] 
            cmap = make_cmap(colors, bit=True)
            cmap.set_under(color="LightGrey")
        elif col_type in cm.__dict__.keys():
            cmap=plt.get_cmap(col_type)    
        else:
            col_names = col_type
            cmap = make_cmap(col_names, position = position, log = log, bit=True)
    except:
        print('Could not find or create appropriate colormap')
        print('Assigning default one (jet)')
        cmap=plt.get_cmap('jet')  

    return cmap
              
             
             
def make_cmap(colors, position=None, bit=False, log=False):
     '''
     make_cmap takes a list of tuples which contain RGB values. The RGB
     values may either be in 8-bit [0 to 255] (in which bit must be set to
     True when called) or arithmetic [0 to 1] (default). make_cmap returns
     a cmap with equally spaced colors.
     Arrange your tuples so that the first color is the lowest value for the
     colorbar and the last is the highest.
     position contains values from 0 to 1 to dictate the location of
     each color.
     '''

     bit_rgb = np.linspace(0,1,256)
     if position == None:
         if not log:
             position = np.linspace(0,1,len(colors))
         else:
             position = (np.logspace(0,np.log10(11),len(colors))-1)/10.      
             # This is to account for issue where position[-1] = 0.99999999...
             position[0] = 0.
             position[-1] = 1. 
     else:
         if len(position) != len(colors):
             raise ValueError("position length must be the same as colors")
         elif position[0] != 0 or position[-1] != 1:
             raise ValueError("position must start with 0 and end with 1")
     if bit:
         for i in range(len(colors)):
             colors[i] = (bit_rgb[colors[i][0]],bit_rgb[colors[i][1]],bit_rgb[colors[i][2]])
     cdict = {'red':[], 'green':[], 'blue':[]}
     for pos, color in zip(position, colors):
         cdict['red'].append((pos, color[0], color[0]))
         cdict['green'].append((pos, color[1], color[1]))
         cdict['blue'].append((pos, color[2], color[2]))

     cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
     return cmap
