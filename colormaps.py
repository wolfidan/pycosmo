# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:30 2015

@author: wolfensb
"""

import numpy as np
import matplotlib.pyplot as plt

def get_colormap(col_type):
    try:
        if isinstance(col_type, basestring):
            name = col_type
            if name == 'earth':
                cmap=plt.get_cmap('gist_earth')
            elif name == 'terrain':
                cmap=plt.get_cmap('terrain')        
            elif name == 'jet':
                cmap=plt.get_cmap('jet')        
            elif name == 'temp':
                cmap=plt.get_cmap('coolwarm')
            elif name == 'press':
                cmap=plt.cm.RdBu_r
            elif name == 'precip':
                colors = [(43,66,181), (67,222,139), (245,245,45), (252,45,45)] 
                cmap=make_cmap(colors, bit=True)
                cmap.set_under(color="LightGrey")
            elif name == 'grays':
                cmap=plt.get_cmap('Greys')
            elif name == 'blues':
                cmap=plt.get_cmap('Blues')   
            elif name == 'reds':
                cmap=plt.get_cmap('Reds')  
            elif name == 'greens':
                cmap=plt.get_cmap('Greens')    
        else:
            col_names = col_type
            print col_names
            cmap=make_cmap(col_names, bit=True)
    except:
        print 'Could not find or create appropriate colormap'
        print 'Assigning default one (jet)'
        cmap=plt.get_cmap('jet')  

    return cmap
              
             
             
def make_cmap(colors, position=None, bit=False):
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
     import matplotlib as mpl
     import numpy as np
     from matplotlib.colors import LogNorm

     bit_rgb = np.linspace(0,1,256)
     if position == None:
         position = np.linspace(0,1,len(colors))
     else:
         if len(position) != len(colors):
             sys.exit("position length must be the same as colors")
         elif position[0] != 0 or position[-1] != 1:
             sys.exit("position must start with 0 and end with 1")
     if bit:
         for i in range(len(colors)):
             colors[i] = (bit_rgb[colors[i][0]],
                          bit_rgb[colors[i][1]],
                          bit_rgb[colors[i][2]])
     cdict = {'red':[], 'green':[], 'blue':[]}
     for pos, color in zip(position, colors):
         cdict['red'].append((pos, color[0], color[0]))
         cdict['green'].append((pos, color[1], color[1]))
         cdict['blue'].append((pos, color[2], color[2]))

     cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
     return cmap
