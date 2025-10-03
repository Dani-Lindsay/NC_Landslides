#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

ts_cluster_lat	ts_cluster_lon
    # Dry seasons
    "ts_dry1_vel_myr", "ts_dry1_err_myr",
    "ts_dry2_vel_myr", "ts_dry2_err_myr",

    # Earthquake 1 post-event windows
    "ts_eq1_3month_vel_myr", "ts_eq1_3month_err_myr",
    "ts_eq1_6month_vel_myr", "ts_eq1_6month_err_myr",
    "ts_eq1_12month_vel_myr", "ts_eq1_12month_err_myr",

    # Earthquake 2 post-event windows
    "ts_eq2_3month_vel_myr", "ts_eq2_3month_err_myr",
    "ts_eq2_6month_vel_myr", "ts_eq2_6month_err_myr",
    "ts_eq2_12month_vel_myr", "ts_eq2_12month_err_myr",

    # Whole time series
    "ts_linear_vel_myr", "ts_linear_err_myr",


'''

import os, re, glob
import numpy as np
import pandas as pd
import pygmt
from NC_Landslides_paths import *  # provides fig_dir etc.

# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 2  # cm/yr threshold for inclusion (>= in either year)

# Region/panel sizing
min_lon, max_lon = -124.5, -122.5
min_lat, max_lat =  39.1,   42.3
region = f"{min_lon}/{max_lon}/{min_lat}/{max_lat}"
panel_width = "M6c"      # each panel width
panel_xshift = "9.5c"      # horizontal spacing between panels

# -------------------------
# Load data
# -------------------------
#df = pd.read_csv("/Volumes/Seagate/NC_Landslides/Data_2/LS_Timeseries/landslides_wy_pga_precip_summary.csv")
df = pd.read_csv(os.path.join(ts_final_dir, "final_selection_only.csv"))

# -------------------------
# Build the 3-panel figure
# -------------------------
fig = pygmt.Figure()

pygmt.config(FONT=12, FONT_TITLE=12.5, MAP_HEADING_OFFSET=0.1, MAP_TITLE_OFFSET= "-7p",
             FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

with fig.subplot(nrows=1, ncols=3, figsize=("19c", "12.5c"), autolabel="a)",
    sharex="b", sharey="l", frame=["xa", "ya", "WSrt"]):

    # Scatter plots
    fig.basemap(region=region, projection=panel_width, frame=["WSrt", "xa", "ya"], panel=True)
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, projection=panel_width, frame=[f"lrtb+tSlope"])
    pygmt.makecpt(cmap="batlow", series=[df["ls_mean_slope"].min(),df["ls_mean_slope"].max()])
    fig.plot(region=region, projection=panel_width, 
             x=df["Lon"], y=df["Lat"], fill=df["ls_mean_slope"],
             style="c0.15c", cmap=True, pen="gray25")
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        fig.colorbar(cmap=True, position="jBL+o0.3c/0.3c+w3c/0.4c", frame=["xaf+lSlope", "ya"], region=region, projection=panel_width)

        
    # Scatter plots
    fig.basemap(region=region, projection=panel_width, frame=["wSrt", "xa", "ya"], panel=True)
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, projection=panel_width, frame=[f"lrtb+tAspect"])
    pygmt.makecpt(cmap="romaO", series=[0,360])
    fig.plot(region=region, projection=panel_width, 
             x=df["Lon"], y=df["Lat"], fill=df["ls_mean_aspect"],
             style="c0.15c", cmap=True, pen="gray25")
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        fig.colorbar(cmap=True, position="jBL+o0.3c/0.3c+w3c/0.4c", frame=["xaf+lAspect", "ya"], region=region, projection=panel_width)
    
    # Scatter plots
    fig.basemap(region=region, projection=panel_width, frame=["wSrt", "xa", "ya"], panel=True, )
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, projection=panel_width, frame=[f"lrtb+tHeight"])
    pygmt.makecpt(cmap="bamako", series=[0, 1700])
    fig.plot(region=region, projection=panel_width, 
             x=df["Lon"], y=df["Lat"], fill=df["ls_mean_height"],
             style="c0.15c", cmap=True, pen="gray25")
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        fig.colorbar(cmap=True, position="jBL+o0.3c/0.3c+w3c/0.4c", frame=["xaf+lHeight", "ya"], region=region, projection=panel_width)
    
fig.savefig(fig_dir+"/Sup_map_aspect_slope_height.png", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(fig_dir+"/Sup_aspect_slope_height.jpeg", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(fig_dir+"/Sup_aspect_slope_height.pdf", crop=True, anti_alias=True, show=False)

fig.show()

