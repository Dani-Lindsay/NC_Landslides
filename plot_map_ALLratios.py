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
df = pd.read_csv(common_paths["ls_disp_pga_precip"])

label = "dry"
# Absolute velocities in cm/yr (for filters + scatter)
df["vel_dry1"] = np.abs(df["ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["ts_dry2_vel_myr"] * 100)
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

# Filter for significant motion (>= threshold in either year)
df = df[(df["vel_dry1"] > vel_min_threshold) | (df["vel_dry2"] > vel_min_threshold)].copy()

# Convenience references
lon_col = "ts_cluster_lon"
lat_col = "ts_cluster_lat"

# -------------------------
# Grouping functions
# -------------------------
def group_by_vel_ratio(r):
    if r < 1/5:
        return "Much Slower"
    elif 1/5 <= r < 0.83:
        return "Slower"
    elif 0.83 <= r <= 1.2:
        return "Similar"
    elif 1.2 < r <= 5:
        return "Faster"
    else:
        return "Much Faster"

def group_by_pga_ratio(r):
    # Option 2 thresholds chosen earlier
    if r < 0.6:
        return "Much Lower"
    elif 0.6 <= r < 0.83:
        return "Lower"
    elif 0.83 <= r <= 1.2:
        return "Similar"
    elif 1.2 < r <= 3:
        return "Higher"
    else:
        return "Much Higher"

def group_by_precip_ratio(r):
    # 3 bins (data span ≈ 1.0–1.9)
    if r <= 1.1:
        return "Similar"
    elif 1.1 < r <= 1.5:
        return "Higher"
    else:
        return "Much Higher"

# Make the three categorical columns
df["vel_group"]    = df["vel_ratio"].apply(group_by_vel_ratio)
df["pga_ratio"]    = df["wy23_vs_wy22_pga_ratio"]
df["pga_group"]    = df["pga_ratio"].apply(group_by_pga_ratio)
df["precip_ratio"] = df["wy23_vs_wy22_rain_ratio"]
df["precip_group"] = df["precip_ratio"].apply(group_by_precip_ratio)

# -------------------------
# Palettes / orders
# -------------------------
# Velocity categories (blue→yellow→red like before)
vel_order  = ["Much Slower", "Slower", "Similar", "Faster", "Much Faster"]
vel_colors = {
    "Much Slower": "#3b4cc0",
    "Slower":      "#a6bddb",
    "Similar":     "#fefcbf",
    "Faster":      "#f4a582",
    "Much Faster": "#b2182b",
}

# PGA categories (reversed Spectral so higher=red)
pga_order  = ["Much Lower", "Lower", "Similar", "Higher", "Much Higher"]
pga_palette = pygmt.makecpt(cmap="roma", series=[0, 4], continuous=True)  # we won't use this directly; keep for consistency

# Pick 5 distinct colors from reversed Spectral
# (hexes pulled from matplotlib's Spectral_r 5-point sampling)
pga_colors = {
    "Much Lower":  "#3b4cc0",
    "Lower":       "#a6bddb",
    "Similar":     "#fefcbf",
    "Higher":      "#f4a582",
    "Much Higher": "#b2182b",
}

# Precip categories (3 colors from same reversed Spectral for consistency)
precip_order  = ["Similar", "Higher", "Much Higher"]
precip_colors = {
    "Similar":     "#fefcbf",
    "Higher":      "#f4a582",
    "Much Higher": "#b2182b",
}



# -------------------------
# Helper to draw a categorical map panel
# -------------------------
def plot_group_panel(fig, data, group_col, order, colors, title=None, add_quakes=False):
    # Base map
    fig.coast(region=region, projection=panel_width, frame=[f"lrtb+t{title}"],
              shorelines=True, lakes=False, borders="2/thin")

    # Optional focal mechanisms (only once if desired)
    if add_quakes:
        focal_mechanism = dict(strike=245, dip=80, rake=20, magnitude=6.4)
        fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.423, latitude=40.525, depth=17.9, region=region, projection=panel_width)
        focal_mechanism = dict(strike=209, dip=81, rake=10, magnitude=6.2)
        fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.298, latitude=40.390, depth=27.0, region=region, projection=panel_width)

    # Plot each group (ordered)
    for g in order:
        sub = data[data[group_col] == g]
        if sub.empty:
            continue
        fig.plot(
            region=region, projection=panel_width,
            x=sub[lon_col],
            y=sub[lat_col],
            style="c0.15c",
            fill=colors[g],
            pen="gray25",
            #label=g
        )

    # Legend
    #fig.legend(position="JBL+jBL+o0.2c", box=False, region=region, projection=panel_width)

    # Title
    #if title:
    #    fig.text(text=title, font="12p,Helvetica-Bold,black", justify="TR", position="TR", offset="-0.2/-0.2c", region=region, projection=panel_width)


# -------------------------
def plot_group_scatter(fig, data, group_col, order, colors, title=None, add_quakes=False):
    # Plot each group (ordered)
    fig.plot(x=[0,20], y=[0,20], pen="gray25", region=scatter_region, projection=scatter_proj)
    
    for g in order:
        sub = data[data[group_col] == g]
        if sub.empty:
            continue
        fig.plot(
            region=scatter_region, projection=scatter_proj,
            x=sub['vel_dry1'],
            y=sub['vel_dry2'],
            style="c0.15c",
            fill=colors[g],
            pen="gray25",
            label=g
        )
        

    # Legend
    fig.legend(position="JTR+jTR+o0.05c", box="+gwhite", region=scatter_region, projection=scatter_proj, transparency=0.5)
    fig.legend(position="JTR+jTR+o0.05c", box=False, region=scatter_region, projection=scatter_proj)

# -------------------------
# Build the 3-panel figure
# -------------------------

scatter_region = [0,20,0,20]
scatter_proj = "X6c"

fig = pygmt.Figure()

pygmt.config(FONT=12, FONT_TITLE=12.5, MAP_HEADING_OFFSET=0.1, MAP_TITLE_OFFSET= "-7p",
             FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

with fig.subplot(
    nrows=1,
    ncols=3,
    figsize=("19c", "6c"),  # width of 15 cm, height of 6 cm
    autolabel="e)",
    margins=["0.3c", "0.2c"],  # horizontal 0.3 cm and vertical 0.2 cm margins
    #title="My Subplot Heading",
    sharex="b",  # shared x-axis on the bottom side
    sharey="l",  # shared y-axis on the left side
    frame=["xaf+l2022 LOS Velocity (cm/yr)", "ya+l2023 LOS Velocity (cm/yr)", "WSrt"],
):
    # Scatter plots
    fig.basemap(region=scatter_region, projection=scatter_proj, frame=["WSrt", "xa", "ya"], panel=True)
    
    plot_group_scatter(
        fig, df, group_col="vel_group", order=vel_order, colors=vel_colors,
        title="Velocity ratio groups", add_quakes=True
    )
    
    fig.basemap(region=scatter_region, projection=scatter_proj, frame=["wSrt", "xa", "ya"], panel=True)
    plot_group_scatter(
        fig, df, group_col="pga_group", order=pga_order, colors=pga_colors,
        title="PGA ratio groups", add_quakes=True
    )
    
    fig.basemap(region=scatter_region, projection=scatter_proj, frame=["wSrt", "xa", "ya"], panel=True)
    plot_group_scatter(
        fig, df, group_col="precip_group", order=precip_order, colors=precip_colors,
        title="Precipitation ratio groups"
    )
    
fig.shift_origin(xshift="0c", yshift="7c")

with fig.subplot(
    nrows=1,
    ncols=3,
    figsize=("19c", "12.5c"),  # width of 15 cm, height of 6 cm
    autolabel="a)",
    margins=["0.3c", "0.2c"],  # horizontal 0.3 cm and vertical 0.2 cm margins
    #title="My Subplot Heading",
    sharex="b",  # shared x-axis on the bottom side
    sharey="l",  # shared y-axis on the left side
    frame=["WSrt"],
):
    fig.basemap(region=region, projection=panel_width, frame=["WSrt", "xa", "ya"], panel=True)
    plot_group_panel(
        fig, df, group_col="vel_group", order=vel_order, colors=vel_colors,
        title="Velocity ratio groups", add_quakes=True
    )
    
    fig.basemap(region=region, projection=panel_width, frame=["wSrt", "xa", "ya"], panel=True)
    plot_group_panel(
        fig, df, group_col="pga_group", order=pga_order, colors=pga_colors,
        title="PGA ratio groups", add_quakes=True
    )
    
    fig.basemap(region=region, projection=panel_width, frame=["wSrt", "xa", "ya"], panel=True)
    plot_group_panel(
        fig, df, group_col="precip_group", order=precip_order, colors=precip_colors,
        title="Precipitation ratio groups"
    )
    
    
fig.shift_origin(xshift="20c", yshift="0c")

# Plot DEM
fig.basemap(region=region, projection=panel_width, frame=["wSrt", "xa", "ya"])
fig.coast(shorelines=True,lakes=False, borders="2/thin")

# Plot Faults
# Loop through and plot each fault file
# Plot Faults
grd_file_list = sorted(glob.glob(os.path.join("/Volumes/Seagate/NC_Landslides/Timeseries_2","y*_box", "geo/geo_velocity_msk.grd")))

pygmt.makecpt(cmap="vik", series=[-0.05,0.05])
for grd_file in grd_file_list:
    fig.grdimage(grid=grd_file, projection=panel_width, region=region, nan_transparent=True)
    
#for box_file in common_paths["box_poly"]:
#    fig.plot(data=box_file, pen="0.5p,black", transparency=50)
      
with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    pygmt.makecpt(cmap="vik", series=[-5,5])
    fig.colorbar(cmap=True, position="jBL+o0.5c/0.3c+w3c/0.4c", frame=["xaf", "y+lcm/yr"])
    #fig.colorbar(position="jBL+o0.4c/0.4c+w4c/0.4c", frame=["xaf", "y+lmm/yr"],)

fig.text(text="d)", position="TL", offset="0.1/-0.1c", font="12p,Helvetica,black", justify="TL",)

focal_mechanism = dict(strike=245, dip=80, rake=20, magnitude=6.4)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.423, latitude=40.525, depth=17.9, region=region, projection=panel_width)
focal_mechanism = dict(strike=209, dip=81, rake=10, magnitude=6.2)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.298, latitude=40.390, depth=27.0, region=region, projection=panel_width)


min_lon, max_lon = -124.5, -123.4
min_lat, max_lat =  40.2,   41.03

fig.plot(x = [min_lon, min_lon, max_lon, max_lon, min_lon], 
         y = [min_lat, max_lat, max_lat, min_lat, min_lat], 
         pen="0.6p,black,--", transparency=0)



fig.basemap(frame=["wSrt", "xa", "ya"], map_scale="jTR+w50k+o0.4/0.4c", projection=panel_width, region=[region])


fig.shift_origin(xshift="0c", yshift="-7c")

# Region/panel sizing
min_lon, max_lon = -124.5, -123.4
min_lat, max_lat =  40.2,   41.03
region = f"{min_lon}/{max_lon}/{min_lat}/{max_lat}"

fig.basemap(region=region, projection=panel_width, frame=["wSrt", "xa", "ya"])
fig.coast(shorelines=True,lakes=False, borders="2/thin")

#grd_file_list = sorted(glob.glob(os.path.join("/Volumes/Seagate/NC_Landslides/Timeseries_2","y*_box", "geo/geo_velocity_msk.grd")))

pygmt.makecpt(cmap="vik", series=[-0.05,0.05])
for grd_file in grd_file_list:
    fig.grdimage(grid=grd_file, projection=panel_width, region=region, nan_transparent=True)
          
fig.text(text="f)", position="TL", offset="0.1/-0.1c", font="12p,Helvetica,black", justify="TL",)

focal_mechanism = dict(strike=245, dip=80, rake=20, magnitude=6.4)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.423, latitude=40.525, depth=17.9, region=region, projection=panel_width)
focal_mechanism = dict(strike=209, dip=81, rake=10, magnitude=6.2)
fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.298, latitude=40.390, depth=27.0, region=region, projection=panel_width)

fig.basemap(frame=["wSrt", "xa", "ya"], map_scale="jTR+w20k+o0.4/0.4c", projection=panel_width, region=[region])

    
# Save
out_base = f"{fig_dir}/Fig_6_Maps_Scatter_vel_pga_precip_minvel{vel_min_threshold}_{label}"
fig.savefig(out_base + ".png", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(out_base + ".jpeg", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(out_base + ".pdf", crop=True, dpi=300, anti_alias=True, show=False)

fig.show()

