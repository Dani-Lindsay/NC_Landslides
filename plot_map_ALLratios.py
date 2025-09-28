#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
import numpy as np
import pandas as pd
import pygmt
from NC_Landslides_paths import *  # provides fig_dir etc.

# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 1.5  # cm/yr threshold for inclusion (>= in either year)

# Region/panel sizing
min_lon, max_lon = -124.5, -122.5
min_lat, max_lat =  39.1,   42.3
region = f"{min_lon}/{max_lon}/{min_lat}/{max_lat}"
panel_width = "M8c"      # each panel width
panel_xshift = "9.5c"      # horizontal spacing between panels

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv")

# Absolute velocities in cm/yr (for filters + scatter)
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

# Filter for significant motion (>= threshold in either year)
df = df[(df["vel_dry1"] > vel_min_threshold) | (df["vel_dry2"] > vel_min_threshold)].copy()

# Convenience references
lon_col = "meta__ts_cluster_lon"
lat_col = "meta__ts_cluster_lat"

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
df["pga_ratio"]    = df["support_params/wy23_vs_wy22_pga_ratio"]
df["pga_group"]    = df["pga_ratio"].apply(group_by_pga_ratio)
df["precip_ratio"] = df["support_params/wy23_vs_wy22_rain_ratio"]
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
    "Much Lower":  "#3288bd",
    "Lower":       "#99d594",
    "Similar":     "#e6f598",
    "Higher":      "#fdae61",
    "Much Higher": "#d53e4f",
}

# Precip categories (3 colors from same reversed Spectral for consistency)
precip_order  = ["Similar", "Higher", "Much Higher"]
precip_colors = {
    "Similar":     "#e6f598",
    "Higher":      "#fdae61",
    "Much Higher": "#d53e4f",
}

# -------------------------
# Helper to draw a categorical map panel
# -------------------------
def plot_group_panel(fig, data, group_col, order, colors, title=None, add_quakes=False):
    # Base map
    fig.coast(region=region, projection=panel_width,
              frame=["WSrt", "xa", "ya"], shorelines=True, lakes=False, borders="2/thin")

    # Optional focal mechanisms (only once if desired)
    if add_quakes:
        focal_mechanism = dict(strike=245, dip=80, rake=20, magnitude=6.4)
        fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.423, latitude=40.525, depth=17.9)
        focal_mechanism = dict(strike=209, dip=81, rake=10, magnitude=6.2)
        fig.meca(spec=focal_mechanism, scale="0.3c", longitude=-124.298, latitude=40.390, depth=27.0)

    # Plot each group (ordered)
    for g in order:
        sub = data[data[group_col] == g]
        if sub.empty:
            continue
        fig.plot(
            x=sub[lon_col],
            y=sub[lat_col],
            style="c0.15c",
            fill=colors[g],
            pen="black",
            label=g
        )

    # Legend
    fig.legend(position="JBL+jBL+o0.2c", box=False)

    # Title
    if title:
        fig.text(text=title, font="12p,Helvetica-Bold,black", justify="TR", position="TR", offset="-0.2/-0.2c")

# -------------------------
# Build the 3-panel figure
# -------------------------


fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1,
             PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

# Panel 1: Velocity ratio groups
plot_group_panel(
    fig, df, group_col="vel_group", order=vel_order, colors=vel_colors,
    title="Velocity ratio groups", add_quakes=True
)

# Panel 2: PGA ratio bins
fig.shift_origin(xshift=panel_xshift, yshift="0c")
plot_group_panel(
    fig, df, group_col="pga_group", order=pga_order, colors=pga_colors,
    title="PGA ratio groups", add_quakes=True
)

# Panel 3: Precipitation ratio bins
fig.shift_origin(xshift=panel_xshift, yshift="0c")
plot_group_panel(
    fig, df, group_col="precip_group", order=precip_order, colors=precip_colors,
    title="Precipitation ratio groups"
)

# Save
out_base = f"{fig_dir}/Fig_6_LocationMaps_vel_pga_precip_minvel{vel_min_threshold}"
fig.savefig(out_base + ".png", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(out_base + ".jpeg", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(out_base + ".pdf", crop=True, dpi=300, anti_alias=True, show=False)

fig.show()
