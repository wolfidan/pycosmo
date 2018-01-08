import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    from mpl_toolkits.basemap import Basemap
    NO_BASEMAP=False
except:
    NO_BASEMAP=True


BASEMAPS={} # Dictionary of basemaps


def plot(var, options={},basemap='',overlay_options={}):

    if isinstance(var,list):
        _overlay(var,options,overlay_options)
        return

    fig={} # The output

    if var.dim == 3:
        print('No 3D plotting functions are implemented yet...sorry')
        print('Returning empty handle')
        return

    if 'cmap' not in options.keys():
        options['cmap']=get_colormap('jet')
    if 'scale' not in options.keys():
        options['scale']='linear'
    if 'filled' not in options.keys():
        options['filled'] = True
    if 'alt_coordinates' not in options.keys() or not var.slice_type in \
        ['lat','lon','latlon','PPI','RHI']:
        options['alt_coordinates'] = False
    if 'levels' not in options.keys():
        if options['filled']: num_levels=25
        else: num_levels=15
        if options['scale']=='log':
            options['levels']=np.logspace(np.nanmin(var[:].ravel()),
                                np.nanmax(var[:].ravel()),num_levels)
        else:
            options['levels']=np.linspace(np.nanmin(var[:].ravel()),
                                np.nanmax(var[:].ravel()),num_levels)
    if 'no_colorbar' not in options.keys():
        options['no_colorbar']=False
    if 'cargs' not in options.keys():
        options['cargs']={}
    if 'cbar_title' not in options.keys():
        options['cbar_title']= var.attributes['units']


    coord_names=var.coordinates.keys()
    if 'lat_2D' in coord_names and 'lon_2D' in coord_names and not NO_BASEMAP:
        spatial=True
    else: spatial=False

    mask = np.isnan(var.data)

    if spatial:
        m = _get_basemap(var,basemap = basemap)

        x, y = m(var.coordinates['lon_2D'],
                 var.coordinates['lat_2D']) # compute map proj coordinates

        m.contourf(x,y,mask,levels=[0.0,0.1,1],colors=['white','Grey'])
        if options['filled']:
            if options['scale'] == 'log':
                vmin=min(options['levels'])
                vmax=max(options['levels'])
                data_no_zero=var.data
                data_no_zero[data_no_zero<vmin]=0.00000001 # Hack to use log scale (won't be plotted)
                CS = m.contourf(x,y,var.data, cmap=options['cmap'],
                              levels=options['levels'], vmin=vmin, vmax=vmax,
                              norm=LogNorm(vmin=vmin, vmax=vmax),
                              alpha = options['alpha'], **options['cargs'])
            else:
                CS = m.contourf(x,y,var.data, cmap=options['cmap'],
                              levels=options['levels'],extend='max',
                              vmin=min(options['levels']),
                              alpha = options['alpha'],**options['cargs'])
        else:

            mask = mask.astype(float)
            CS=m.contour(x,y,var.data, cmap=options['cmap'],
                         levels=options['levels'],extend='max',
                         alpha = options['alpha'],
                         vmin=min(options['levels']),**options['cargs'])
            plt.clabel(CS, inline=1, fontsize=9)

        fig['basemap']=m
    else:
        if options['alt_coordinates']:
            try:
                if var.slice_type in ['lat','lon','latlon']:
                    y = var.attributes['z-levels']
                    x = var.coordinates[coord_names[1]]
                    x = np.tile(x, (len(y),1))
                elif var.slice_type == 'PPI':
                    y = var.attributes['lat_2D']
                    x = var.attributes['lon_2D']
                elif var.slice_type == 'RHI':
                    x = var.attributes['dist_ground']
                    y = var.attributes['altitude']
            except:
                print('Could not plot on altitude levels, plotting on model'+\
                      ' levels instead...')
                options['plot_altitudes'] = False


        if not options['alt_coordinates']:
            x=var.coordinates[coord_names[1]]
            y=var.coordinates[coord_names[0]]
        plt.contourf(x,y,mask, levels=[0.0,0.1,1],colors=['white','Grey'])
        if options['filled']:
            if options['scale'] == 'log':
                 vmin=min(options['levels'])
                 vmax=max(options['levels'])
                 data_no_zero=var.data
                 data_no_zero[data_no_zero<vmin]=0.00000001 # Hack to use log scale (won't be plotted)

                 CS=plt.contourf(x,y,var.data, cmap=options['cmap'],
                                 levels=options['levels'], vmin=vmin,
                                 vmax=vmax, norm=LogNorm(vmin=vmin,
                                 vmax=vmax), alpha = options['alpha'],
                                 **options['cargs'])
            else:
                 CS=plt.contourf(x,y,var.data, cmap=options['cmap'],
                                 levels=options['levels'], extend='max'
                                 ,vmin=min(options['levels']),
                                 alpha = options['alpha'],**options['cargs'])
        else:
             mask=mask.astype(float)
             CS=plt.contour(x,y,var.data, cmap=options['cmap'],
                            levels=options['levels'],extend='max',
                            vmin=min(options['levels']),
                            alpha = options['alpha'],**options['cargs'])
             plt.clabel(CS, inline=1, fontsize=9)
             plt.xlabel(coord_names[1])
             plt.ylabel(coord_names[0])

    plt.title(var.attributes['long_name'].capitalize()+' ['+var.attributes['units']+']')

    if not options['no_colorbar']:
        if options['filled']:
            cbar=plt.colorbar(fraction=0.046, pad=0.04, label=options['cbar_title'])
            cbar.set_ticks(options['levels'])
            cbar.set_ticklabels(_format_ticks(options['levels'],decimals=2))
        else:
            ax=plt.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            for pc in CS.collections:
                proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_edgecolor()[0]) for pc in CS.collections]
            lgd=plt.legend(proxy, _format_ticks(options['levels'],decimals=2),loc='center left', bbox_to_anchor=(1, 0.6), title=options['cbar_title'])
            ax.add_artist(lgd)

    if options['alt_coordinates']:
        if var.slice_type in ['lat','lon','latlon']:
            plt.ylabel('Altitude [m]')
        elif var.slice_type == 'PPI':
            plt.ylabel('lat [deg]')
            plt.xlabel('lon [deg]')
        elif var.slice_type == 'RHI':
            plt.ylabel('altitude [m]')
            plt.xlabel('distance [m]')

        plt.gca().set_axis_bgcolor('Gray')

    fig['cont_handle']=CS
    fig['fig_handle']=plt.gcf()
    del options

    return fig


def _overlay(list_vars, var_options=[{},{}], overlay_options={}):

    overlay_options_keys=overlay_options.keys()

    if 'labels' not in overlay_options_keys:
        overlay_options['labels']=[var.name for var in list_vars]
    if 'label_position' not in overlay_options_keys:
        overlay_options['label_position']='right'

    plt.hold(False)
    n_vars=len(list_vars)
    offsets=np.linspace(0.9,0.1,n_vars)
    basemap=''
    for idx, var in enumerate(list_vars):
        plt.hold(True)
        opt = var_options[idx]
        opt['no_colorbar'] = True
        fig = plot(var,var_options[idx], basemap)
        ax = plt.gca()
        box = ax.get_position()

        filled = opt['filled']
        if filled:
            for pc in fig['cont_handle'].collections:
                proxy = [plt.Rectangle((0,0),1,1, fc = pc.get_facecolor()[0]) for pc in fig['cont_handle'].collections]
        else:
            for pc in fig['cont_handle'].collections:
                proxy = [plt.Rectangle((0,0),1,1, fc = pc.get_edgecolor()[0]) for pc in fig['cont_handle'].collections]

        if overlay_options['label_position'] == 'right':
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd=plt.legend(proxy, _format_ticks(opt['levels'],decimals=2),
                           loc='center left', bbox_to_anchor=(1, offsets[idx]),
                           title=overlay_options['labels'][idx])
        elif overlay_options['label_position'] == 'left':
            ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            lgd=plt.legend(proxy, _format_ticks(opt['levels'],decimals=2),
                           loc='center right', bbox_to_anchor=(-0.05, offsets[idx]),
                           title=overlay_options['labels'][idx])
        elif overlay_options['label_position'] == 'top':
            ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
            lgd=plt.legend(proxy, _format_ticks(opt['levels'],decimals=2),
                           loc='lower center', bbox_to_anchor=(offsets[idx],1.05),
                           title=overlay_options['labels'][idx])
        elif overlay_options['label_position'] == 'bottom':
            ax.set_position([box.x0, box.y0+0.05*box.height, box.width, box.height*0.8])
            lgd=plt.legend(proxy, _format_ticks(opt['levels'],decimals=2),
                           loc='upper center', bbox_to_anchor=(offsets[idx],-0.05),
                           title=overlay_options['labels'][idx])
        ax.add_artist(lgd)
    return fig

def _format_ticks(labels,decimals=2):
    labels_f=labels
    for idx, val in enumerate(labels):
        if int(val) == val:  labels_f[idx]=val
        else: labels_f[idx]=round(val,decimals)
    return labels_f


def make_colorbar(fig,orientation='horizontal',label=''):

    if orientation == 'horizontal':
        cbar_ax = fig['fig_handle'].add_axes([0.2, 0,0.6,1])
        axins = inset_axes(cbar_ax,
               width="100%", # width = 10% of parent_bbox width
               height="5%", # height : 50%
               loc=10,
               bbox_to_anchor=(0, -0.01, 1 , 0.15),
               bbox_transform=cbar_ax.transAxes,
               borderpad=0,
               )
    else:
        cbar_ax = fig['fig_handle'].add_axes([0, 0.2,1,0.6])
        axins = inset_axes(cbar_ax,
               width="5%", # width = 10% of parent_bbox width
               height="100%", # height : 50%
               loc=6,
               bbox_to_anchor=(1.01, 0, 0.15, 1),
               bbox_transform=cbar_ax.transAxes,
               borderpad=0,
               )
    cbar_ax.get_xaxis().tick_bottom()
    cbar_ax.axes.get_yaxis().set_visible(False)
    cbar_ax.axes.get_xaxis().set_visible(False)
    cbar_ax.set_frame_on(False)

    cbar=plt.colorbar(cax=axins, orientation=orientation,label=label)

    levels = fig['cont_handle'].levels
    cbar.set_ticks(levels)
    cbar.set_ticklabels(_format_ticks(levels,decimals=2))
    return cbar

def _get_basemap(var, basemap = ''):
    domain=var.attributes['domain_2D']
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
