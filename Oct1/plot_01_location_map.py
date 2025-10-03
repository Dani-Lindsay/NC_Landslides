#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:31:35 2025

@author: daniellelindsay
"""

import pygmt
import pandas as pd
import numpy as np
import os, glob, shutil
from NC_Landslides_paths import *


csv_path = os.path.join(ts_final_dir, "final_selection.csv")
box_df = pd.read_csv('/Volumes/Seagate/NC_Landslides/Inputs/box_corners_all.csv')

# Load data
df = pd.read_csv(csv_path)
ls_lons = df['center_lon'].values
ls_lats = df['center_lat'].values
# ls_area = df['area_m2'].values
# ls_vel  = df['ts_linear_vel_myr'].values*100




# Sort by absolute velocity so largest magnitudes plot last
order = np.argsort(np.abs(ls_lats))

ls_lons = ls_lons[order]
ls_lats = ls_lats[order]
#ls_vel = ls_vel[order]

# Define region of interest 
min_lon=-125.5
max_lon=-121.0
min_lat=38.6
max_lat=42.5

region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M9c"

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


# fig.plot(x=eq_21_lon, y=eq_21_lat, style="a0.4c", pen="0.6p,black", fill="goldenrod")
# fig.plot(x=eq_22_lon, y=eq_22_lat, style="a0.4 c", pen="0.6p,black", fill="dodgerblue2")

pygmt.makecpt(cmap="roma", series=[-10,10], reverse=True)
#fig.plot(x=ls_lons, y=ls_lats, fill=ls_vel, cmap=True, style="c0.15c", pen="black")
fig.plot(x=ls_lons, y=ls_lats, fill="dodgerblue2", style="c0.1c", pen="black")


# Store focal mechanism parameters in a dictionary based on the Aki & Richards
# convention
#mt = dict(t_value=2.017, t_azimuth=14, t_plunge=74, n_value=0.020, n_azimuth=76, n_plunge=251, p_value=-2.037, p_azimuth=1, p_plunge=344, exponent=18)
focal_mechanism = dict(strike=245, dip=80, rake=20, magnitude=6.4)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.423, latitude=40.525, depth=17.9)
fig.text(text="2022", x=-124.423 - 0.05, y=40.525 + 0.05, justify="MR", font="10p,Bold,dodgerblue4")

focal_mechanism = dict(strike=209, dip=81, rake=10, magnitude=6.2)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.298, latitude=40.390, depth=27.0)
fig.text(text="2021", x=-124.298 - 0.05, y=40.390 -0.05 , justify="MR", font="10p,Bold,dodgerblue4")

# focal_mechanism = dict(strike=213, dip=88, rake=13, magnitude=5.4)
# fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-123.971, latitude=40.409, depth=27.0)
# fig.text(text="2023", x=-123.971 + 0.05, y=40.409 -0.05, justify="ML", font="10p,dodgerblue4")
 


# with pygmt.config(
#         FONT_ANNOT_PRIMARY="18p,black", 
#         FONT_ANNOT_SECONDARY="18p,black",
#         FONT_LABEL="18p,black",
#         ):
#     fig.colorbar(cmap=True, position="jBR+o1c/0.3c+w3c/0.4c", frame=["xaf", "y+lcm/yr"])

fig.text(text="a)", position="TL", offset="0.1/-0.1c", font="12p,Helvetica,black", justify="TL",)

# Inset map of frames

# Define region of interest 
sub_min_lon=-126.5
sub_max_lon=-116.0
sub_min_lat=35.0
sub_max_lat=46.0

sub_region="%s/%s/%s/%s" % (sub_min_lon, sub_max_lon, sub_min_lat, sub_max_lat)

fig.basemap(projection="M2.45c", frame=["lSrt"], region=[sub_region])
fig.coast(shorelines=True, land="lightgray", water="white",  borders="2/0.5p,gray15", area_thresh=5000)
fig.plot(data=common_paths['170_2800_frame'] , pen="0.6p,black,--", label='Track 170')
fig.plot(x = [min_lon, min_lon, max_lon, max_lon, min_lon], 
         y = [min_lat, max_lat, max_lat, min_lat, min_lat], 
         pen="0.6p,black", transparency=0)



fig.text(text="California", x=-121.0, y=37.5, justify="CM", font="8p,black", fill="white", transparency=50 )
fig.text(text="Oregon", x=-121.0, y=43.5, justify="CM", font="8p,black",  fill="white", transparency=50 )
fig.text(text="California", x=-121.0, y=37.5, justify="CM", font="8p,black" )
fig.text(text="Oregon", x=-121.0, y=43.5, justify="CM", font="8p,black" )




fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jTR+w50k+o0.4/0.4c", projection=fig_size, region=[region])

fig.shift_origin(xshift="w+0.5c")
# Plot DEM
fig.grdimage(grid=grid, projection=fig_size, frame=["wSrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=region, transparency=40)
fig.coast(shorelines=True,lakes=False, borders="2/thin")

# Plot Faults
# Loop through and plot each fault file
# Plot Faults
grd_file_list = sorted(glob.glob(os.path.join("/Volumes/Seagate/NC_Landslides/Timeseries_2","y*_box", "geo/geo_velocity_msk.grd")))

pygmt.makecpt(cmap="vik", series=[-0.05,0.05])
for grd_file in grd_file_list:
    fig.grdimage(grid=grd_file, projection=fig_size, region=region, nan_transparent=True)
    
for box_file in common_paths["box_poly"]:
    fig.plot(data=box_file, pen="0.5p,black", transparency=50)
    
for wildfire_file in common_paths["wildfire_poly"]:
    fig.plot(data=wildfire_file, pen="0.3p,darkred", transparency=50)
    
    
fig.plot(data=common_paths['pb_file'] , pen="0.8p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")
fig.plot(data=common_paths['pb2_file'] , pen="0.8p,red3", style="f0.5c/0.15c+r+t", fill="red3")
fig.plot(data=common_paths['pb3_file'] , pen="0.8p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")

with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    pygmt.makecpt(cmap="vik", series=[-5,5])
    fig.colorbar(cmap=True, position="jBL+o0.3c/0.3c+w3c/0.4c", frame=["xaf", "y+lcm/yr"])
    #fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)

fig.text(text="b)", position="TL", offset="0.1/-0.1c", font="12p,Helvetica,black", justify="TL",)
fig.basemap(frame=["wSrt", "xa", "ya"], map_scale="jTR+w50k+o0.4/0.4c", projection=fig_size, region=[region])


# fig.shift_origin(xshift="w+0.5c")

# with fig.subplot(
#     nrows=2,
#     ncols=3,
#     figsize=("18c", "8c"),  # width of 15 cm, height of 6 cm
#     autolabel=True,
#     margins=["0.3c", "0.2c"],  # horizontal 0.3 cm and vertical 0.2 cm margins
#     title="My Subplot Heading",
#     sharex="b",  # shared x-axis on the bottom side
#     sharey="l",  # shared y-axis on the left side
#     frame="WSrt",
# ):
#     fig.basemap(region=[0, 10, 0, 10], projection="X?", panel=True)
#     fig.basemap(region=[0, 20, 0, 10], projection="X?", panel=True)
#     fig.basemap(region=[0, 10, 0, 20], projection="X?", panel=True)
#     fig.basemap(region=[0, 20, 0, 20], projection="X?", panel=True)
#     fig.basemap(region=[0, 10, 0, 20], projection="X?", panel=True)
#     fig.basemap(region=[0, 20, 0, 20], projection="X?", panel=True)


fig.savefig(fig_dir+"/Fig_1_Location_map.png", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(fig_dir+"/Fig_1_Location_map.jpeg", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(fig_dir+"/Fig_1_Location_map.pdf", crop=True, anti_alias=True, show=False)

fig.show()