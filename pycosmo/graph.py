import pycosmo.utilities as utilities

import numpy as np
from collections import OrderedDict
from colormaps import get_colormap
from matplotlib.colors import LogNorm
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
                              norm=LogNorm(vmin=vmin, vmax=vmax),**options['cargs'])
            else:
                CS=m.contourf(x,y,var.data, cmap=options['cmap'],levels=options['levels'],extend='max',  
                              vmin=min(options['levels']),**options['cargs'])
        else:
               
            mask = mask.astype(float)
            CS=m.contour(x,y,var.data, cmap=options['cmap'], 
                         levels=options['levels'],extend='max',
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
                                 vmax=vmax),**options['cargs'])
            else:
                 CS=plt.contourf(x,y,var.data, cmap=options['cmap'], 
                                 levels=options['levels'], extend='max'
                                 ,vmin=min(options['levels']),**options['cargs'])
        else:
             mask=mask.astype(float)
             CS=plt.contour(x,y,var.data, cmap=options['cmap'], 
                            levels=options['levels'],extend='max', 
                            vmin=min(options['levels']),**options['cargs'])
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
        
        for pc in fig['cont_handle'].collections:
                proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_edgecolor()[0]) for pc in fig['cont_handle'].collections] 
        
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