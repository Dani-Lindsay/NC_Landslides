#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:31:35 2025

@author: daniellelindsay
"""

import pygmt
import pandas as pd
import numpy as np
from NC_Landslides_paths import *

# Load data
df = pd.read_csv('/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_2/final_selection_only.csv')
ls_lons = df['center_lon'].values
ls_lats = df['center_lat'].values
ls_area = df['area_m2'].values
ls_vel  = df['ts_linear_vel_myr'].values*100

# Sort by absolute velocity so largest magnitudes plot last
order = np.argsort(np.abs(ls_vel))

ls_lons = ls_lons[order]
ls_lats = ls_lats[order]
ls_vel = ls_vel[order]

# Define region of interest 
min_lon=-125.5
max_lon=-122.0
min_lat=38.6
max_lat=42.5

region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M8c"

grid = pygmt.datasets.load_earth_relief(region=region, resolution="15s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=region)

### Begin plotting ###
fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain",)

# Plot DEM
fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=region, transparency=40)
fig.coast(shorelines=True,lakes=False, borders="2/thin")

# Plot Faults
# Loop through and plot each fault file
# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50)
    
fig.plot(data=common_paths['pb_file'] , pen="0.8p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")
fig.plot(data=common_paths['pb2_file'] , pen="0.8p,red3", style="f0.5c/0.15c+r+t", fill="red3")
fig.plot(data=common_paths['pb3_file'] , pen="0.8p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")

#fig.plot(data="/Users/daniellelindsay/Figures/inputdata/wcSlides.gmt", pen="0.3p,purple", fill="purple")

fig.plot(data=common_paths['170_2800_frame'] , pen="1p,black,--", transparency=30, label='Track 170')
# Label Faults
fig.plot(x=-125.1, y=41.7, style="e280/1.5/0.7", fill="white", transparency=30)
fig.text(text="CSZ", x=-125.1, y=41.7, justify="CM", font="10p,red3" , angle=280)

fig.plot(x=-125.1, y=40.65, style="e0/1.5/0.7", fill="white", transparency=30)
fig.text(text="MFZ", x=-125.1, y=40.65, justify="CM", font="10p,red3" )

fig.plot(x=-124.0, y=39.0, style="e310/1.5/0.7", fill="white", transparency=30)
fig.text(text="SAF", x=-124.0, y=39.0, justify="CM", font="10p,red3" , angle=310)


eq_21_lon = -124.298
eq_21_lat = 40.390
eq_21_m = 6.2 
eq_22_lat = 40.525
eq_22_lon = -124.423
eq_22_m = 6.4

#fig.plot(x=eq_21_lon, y=eq_21_lat, style="a0.4c", pen="0.6p,black", fill="goldenrod")
#fig.plot(x=eq_22_lon, y=eq_22_lat, style="a0.4 c", pen="0.6p,black", fill="dodgerblue2")
 
pygmt.makecpt(cmap="roma", series=[-10,10], reverse=True)
fig.plot(
    x=ls_lons,
    y=ls_lats,
    #size=0.2c, # * 2**ls_area,
    fill=ls_vel,
    cmap=True,
    style="c0.15c",
    pen="black",
)

# Inset map of frames

# Define region of interest 
sub_min_lon=-126.5
sub_max_lon=-116.0
sub_min_lat=35.0
sub_max_lat=46.0

sub_region="%s/%s/%s/%s" % (sub_min_lon, sub_max_lon, sub_min_lat, sub_max_lat)

fig.basemap(projection="M2.75c", frame=["lbrt"], region=[sub_region])
fig.coast(shorelines=True, land="lightgray", water="white",  borders="2/0.5p,gray15", area_thresh=5000)
fig.plot(data=common_paths['170_2800_frame'] , pen="0.6p,black,--", label='Track 170')
fig.plot(x = [min_lon, min_lon, max_lon, max_lon, min_lon], 
         y = [min_lat, max_lat, max_lat, min_lat, min_lat], 
         pen="0.6p,black", transparency=0)

fig.text(text="California", x=-121.0, y=37.5, justify="CM", font="8p,black", fill="white", transparency=50 )
fig.text(text="Orgeon", x=-121.0, y=43.5, justify="CM", font="8p,black",  fill="white", transparency=50 )
fig.text(text="California", x=-121.0, y=37.5, justify="CM", font="8p,black" )
fig.text(text="Orgeon", x=-121.0, y=43.5, justify="CM", font="8p,black" )

fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jTR+w50k+o0.4/0.4c", projection=fig_size, region=[region])

fig.savefig(fig_dir+"/Location_map.png", crop=True, anti_alias=True, show=False)
fig.savefig(fig_dir+"/Location_map.pdf", crop=True, anti_alias=True, show=False)

fig.show()