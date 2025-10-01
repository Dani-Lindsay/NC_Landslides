# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_plot(ax, lon, lat, los, polygon):

    lon, lat = np.meshgrid(lon, lat)
    
    im = ax.pcolormesh(lon, lat, los, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='LOS (m)')
    
    ax.plot(polygon[:,0],polygon[:,1], 'k')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Heatmap of LOS values')
    
    return ax

def create_mask(lons, lats, coords):
    path = mpath.Path(coords)
    x, y = np.meshgrid(lons, lats)
    points = np.vstack((x.flatten(), y.flatten())).T
    mask = path.contains_points(points).reshape(x.shape)
    return mask

data_dir = '/Users/andrewmcnab/Downloads/Data'
figure_dir = '/Users/andrewmcnab/Downloads/Figures'
training_dir = '/Users/andrewmcnab/Downloads/Training'

grd_files = glob.glob(os.path.join(data_dir,'*.grd'))

for filename in grd_files:
    event_id = filename.split('/')[-1].split('_')[0]
    print(f'running {event_id}')
    polygon_path = os.path.join(data_dir, f'{event_id}_coords.txt')
    
    if os.path.exists(polygon_path):
        with nc.Dataset(filename, 'r') as grd:
            los = np.array(grd.variables['Annual Precip. (m)'][:])
            lat = np.array(grd.variables['latitude'][:])
            lon = np.array(grd.variables['longitude'][:])
           
        polygon = np.loadtxt(polygon_path, delimiter=',', skiprows=1)
        mask = create_mask(lon, lat, polygon)
        label = np.ones(los.shape) * mask
        label.astype(int)
        
        label_filename = os.path.join(training_dir,'labels',f'label_{event_id}.npy')
        np.save(label_filename,label)
        
        layer_filename = os.path.join(training_dir,'layers',f'layer_{event_id}.npy')
        np.save(layer_filename,los)
        
        fig, (ax1) = plt.subplots(figsize=(8,8))
        ax1 = make_plot(ax1, lon, lat, los, polygon)
        fig_filename = os.path.join(figure_dir,f'{event_id}.png')
        fig.savefig(fig_filename)
        plt.close(fig)
    else:
        
        print(f'\t{event_id} polygon file does not exist')

