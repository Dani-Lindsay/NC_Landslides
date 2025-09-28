#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 18:56:25 2025

@author: daniellelindsay
"""

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


# =========================
# Imports & config
# =========================
import os, re, glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from NC_Landslides_paths import *  # provides fig_dir etc.
import pygmt  # (unused here, but kept if other parts rely on it)

# Load data
df_all = pd.read_csv('/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv')
ls_lons_all = df_all['center_lon'].values
ls_lats_all = df_all['center_lat'].values


# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 1.5     # cm/yr threshold for inclusion (>= in either year)
vel_multiple      = 5     # ratio cut for "Much Faster/Slower"
active_threshold  = 1     # cm/yr threshold for active/inactive annotation

ALPHA = 0.05
B_BOOT = 10_000
SEED_BOOT = 123

# =========================
# Load & prepare data
# =========================
df = pd.read_csv("/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv")

# Absolute velocities in cm/yr
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)

# Keep a copy of low-rate for plotting
df_lowrate = df[(df["vel_dry1"] < vel_min_threshold) | (df["vel_dry2"] < vel_min_threshold)].copy()

# Filter for significant motion (>= vel_min_threshold in either year)
df = df[(df["vel_dry1"] > vel_min_threshold) | (df["vel_dry2"] > vel_min_threshold)].copy()

# Velocity ratio
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

# Grouping function
def categorize_ratio_group(ratio, multiple=vel_multiple):
    if ratio < 1/multiple:
        return "Much Slower"
    elif 1/multiple <= ratio < 0.83:
        return "Slower"
    elif 0.83 <= ratio <= 1.2:
        return "Similar"
    elif 1.2 < ratio <= multiple:
        return "Faster"
    else:
        return "Much Faster"

df["group"] = df["vel_ratio"].apply(categorize_ratio_group)

# Colors
roma_colors = {
    "Much Slower": "#3b4cc0",
    "Slower":      "#a6bddb",
    "Similar":     "#fefcbf",
    "Faster":      "#f4a582",
    "Much Faster": "#b2182b",
    "Low Rate":    "#cccccc"
}

# Load data
ls_lons = df['meta__ts_cluster_lon'].values
ls_lats = df['meta__ts_cluster_lat'].values
ls_vel  = df['meta__ts_linear_vel_myr'].values*100

# Sort by absolute velocity so largest magnitudes plot last
order = np.argsort(np.abs(ls_vel))

ls_lons = ls_lons[order]
ls_lats = ls_lats[order]
ls_vel = ls_vel[order]

# Define region of interest 
min_lon=-124.5
max_lon=-122.5
min_lat=39.1
max_lat=42.3

region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M8c"

### Begin plotting ###
fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",)

# Plot DEM

fig.coast(region=region, projection=fig_size,  frame=["WSrt", "xa", "ya"], shorelines=True,lakes=False, borders="2/thin")
# Store focal mechanism parameters in a dictionary based on the Aki & Richards
# convention
#mt = dict(t_value=2.017, t_azimuth=14, t_plunge=74, n_value=0.020, n_azimuth=76, n_plunge=251, p_value=-2.037, p_azimuth=1, p_plunge=344, exponent=18)
focal_mechanism = dict(strike=245, dip=80, rake=20, magnitude=6.4)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.423, latitude=40.525, depth=17.9)
fig.text(text="2022", x=-124.423 - 0.05, y=40.525 + 0.05, justify="MR", font="10p,Bold,dodgerblue4")

focal_mechanism = dict(strike=209, dip=81, rake=10, magnitude=6.2)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.298, latitude=40.390, depth=27.0)
fig.text(text="2021", x=-124.298 - 0.05, y=40.390 -0.05 , justify="MR", font="10p,Bold,dodgerblue4")


# fig.plot(
#     x=ls_lons_all,
#     y=ls_lats_all,
#     fill=None,
#     style="c0.15c",
#     pen="black",
#     label="Candidate")  # optional: add legend entry)


# Define desired group plotting order
group_order = ["Much Faster", "Much Slower", "Faster", "Slower", "Similar"]


# Instead of makecpt + one fig.plot, do:
for g, color in roma_colors.items():
    sub = df[df["group"] == g]
    if sub.empty:
        continue
    fig.plot(
        x=sub["meta__ts_cluster_lon"],
        y=sub["meta__ts_cluster_lat"],
        fill=color,
        style="c0.15c",
        pen="black",
        label=g  # optional: add legend entry
    )

with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    fig.colorbar(cmap=True, position="jBR+o1c/0.3c+w3c/0.4c", frame=["xaf", "y+lcm/yr"])
    
fig.legend(position="JBL+jBL+o0.2c", box=False)

# Define region of interest 
sub_min_lon=-123.92
sub_max_lon=-123.5
sub_min_lat=40.55
sub_max_lat=40.85

fig.plot(x = [sub_min_lon, sub_min_lon, sub_max_lon, sub_max_lon, sub_min_lon], 
         y = [sub_min_lat, sub_max_lat, sub_max_lat, sub_min_lat, sub_min_lat], 
         pen="0.6p,black,--", transparency=0)

sub_region_1="%s/%s/%s/%s" % (sub_min_lon, sub_max_lon, sub_min_lat, sub_max_lat)


# # Define region of interest 
# sub_min_lon=-123.6
# sub_max_lon=-123.4
# sub_min_lat=41.35
# sub_max_lat=41.6

# fig.plot(x = [sub_min_lon, sub_min_lon, sub_max_lon, sub_max_lon, sub_min_lon], 
#          y = [sub_min_lat, sub_max_lat, sub_max_lat, sub_min_lat, sub_min_lat], 
#          pen="0.6p,black,--", transparency=0)

# sub_region_2="%s/%s/%s/%s" % (sub_min_lon, sub_max_lon, sub_min_lat, sub_max_lat)


# Define region of interest 
sub_min_lon=-124.42
sub_max_lon=-124.0
sub_min_lat=40.3
sub_max_lat=40.6

fig.plot(x = [sub_min_lon, sub_min_lon, sub_max_lon, sub_max_lon, sub_min_lon], 
         y = [sub_min_lat, sub_max_lat, sub_max_lat, sub_min_lat, sub_min_lat], 
         pen="0.6p,black,--", transparency=0)

sub_region_3="%s/%s/%s/%s" % (sub_min_lon, sub_max_lon, sub_min_lat, sub_max_lat)
 

fig.shift_origin(xshift="5.5c", yshift="6c")

fig.basemap(projection="M5.5c", frame=["lSrt"], region=[sub_region_3])
fig.coast(frame=["lbrt"], land="white", shorelines=True,lakes=False, borders="2/thin")

pygmt.makecpt(cmap="vik", series=[-0.08,0.08])
fig.grdimage(grid="/Volumes/Seagate/NC_Landslides/Timeseries_2/y2_x3_box/geo/geo_velocity.grd", nan_transparent=True)
# Instead of makecpt + one fig.plot, do:
for g, color in roma_colors.items():
    sub = df[df["group"] == g]
    if sub.empty:
        continue
    fig.plot(
        x=sub["meta__ts_cluster_lon"],
        y=sub["meta__ts_cluster_lat"],
        fill=None,
        style="c0.15c",
        pen="black",
        label=g  # optional: add legend entry
    )
    
# fig.plot(
#     x=ls_lons_all,
#     y=ls_lats_all,
#     fill=None,
#     style="c0.15c",
#     pen="black",
#     label="Candidate")  # optional: add legend entry)
    

fig.shift_origin(xshift="0c", yshift="5.5c")

fig.basemap(projection="M5.5c", frame=["lSrt"], region=[sub_region_1])
fig.coast(frame=["lbrt"], land="white", shorelines=True,lakes=False, borders="2/thin")

pygmt.makecpt(cmap="vik", series=[-0.08,0.08])
fig.grdimage(grid="/Volumes/Seagate/NC_Landslides/Timeseries_3/y1_x2_box/geo/geo_velocity.grd", nan_transparent=True)
# Instead of makecpt + one fig.plot, do:
for g, color in roma_colors.items():
    sub = df[df["group"] == g]
    if sub.empty:
        continue
    fig.plot(
        x=sub["meta__ts_cluster_lon"],
        y=sub["meta__ts_cluster_lat"],
        fill=None,
        style="c0.15c",
        pen="black",
        label=g  # optional: add legend entry
    )
    
    
# fig.plot(
#     x=ls_lons_all,
#     y=ls_lats_all,
#     fill=None,
#     style="c0.15c",
#     pen="black",
#     label="Candidate")  # optional: add legend entry)
    
with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    pygmt.makecpt(cmap="vik", series=[-8,8])
    fig.colorbar(cmap=True, position="jBR+o2c/-10c+w4c/0.4c", frame=["xaf", "y+lcm/yr"])
    #fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)

fig.savefig(fig_dir+f"/Fig_6_VelRatio_Location_map_minvel{vel_min_threshold}.png", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(fig_dir+f"/Fig_6_VelRatio_Location_map_minvel{vel_min_threshold}.jpeg", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(fig_dir+f"/Fig_6_VelRatio_Location_map_minvel{vel_min_threshold}.pdf", crop=True, anti_alias=True, show=False)

fig.show()