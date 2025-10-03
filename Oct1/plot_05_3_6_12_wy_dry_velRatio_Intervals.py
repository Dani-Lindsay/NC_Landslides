#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity-ratio panels (PyGMT): 1×5 square scatters of V2 vs V1 colored by category.
Author: you; tidied for syntax/logic.
"""

# =========================
# Imports & config
# =========================
import numpy as np
import pandas as pd
import pygmt
import matplotlib as mpl  # only for color names if you need
from NC_Landslides_paths import *  # provides common_paths etc.

# -------------------------
# Parameters
# -------------------------
# thresholds (cm/yr)
VEL_MIN_CM     = 2          # floor to consider velocity reliable (both below -> NaN)
VEL_MULTIPLE   = 4.0           # "Much Faster/Slower" ratio cut
# scatter axes (cm/yr)
AX_MIN, AX_MAX = 0.0, 40.0

# Category definitions and colors
CAT_ORDER = ["No Data","Faster","Slower", "Similar",  "Much Faster", "Much Slower"]
CAT_COLORS = {
    "Low Rate":     "white",
    "Much Slower": "#3b4cc0",
    "Slower":      "#a6bddb",
    "Similar":     "#fefcbf",
    "Faster":      "#f4a582",
    "Much Faster": "#b2182b",
}

# Five velocity windows: (v1_col, v2_col, short_label, panel_title)
PAIRS = [
    ("ts_eq1_3month_vel_myr",  "ts_eq2_3month_vel_myr",  "0_3m",  "0-3 month"),
    ("ts_eq1_3-6month_vel_myr",  "ts_eq2_3-6month_vel_myr",  "3_6m",  "3-6 month"),
    ("ts_eq1_6-9month_vel_myr",  "ts_eq2_6-9month_vel_myr",  "6_9m",  "6-9 month"),
    ("ts_eq1_9-12month_vel_myr", "ts_eq2_9-12month_vel_myr", "9_12m", "9-12 month"),
    ("ts_wy22_vel_myr",        "ts_wy23_vel_myr",        "WY",     "Water Year"),
]

num_groups = 5

# =========================
# Load & prepare data
# =========================
df = pd.read_csv(os.path.join(ts_final_dir, "final_selection_only_with_pga_precip.csv"))

num_ls = len(df)

def categorize_ratio_group(r, multiple=VEL_MULTIPLE):
    """Map a ratio value to a category label."""
    if pd.isna(r) or np.isinf(r):
        return np.nan
    if r < 1.0 / multiple:
        return "Much Slower"
    elif r < 0.83:
        return "Slower"
    elif r <= 1.2:
        return "Similar"
    elif r <= multiple:
        return "Faster"
    else:
        return "Much Faster"

def process_one_period(df, v1_col, v2_col, label, vel_min_cm=VEL_MIN_CM):
    """
    Create columns:
      - f'{label}_v1_cm', f'{label}_v2_cm' (absolute cm/yr)
      - f'ratio_{label}' (V2/V1, NaN if both below vel_min_cm)
      - f'group_{label}' (categorical label; 'Low Rate' if both below floor)
    """
    v1_cm = np.abs(df[v1_col].to_numpy(float) * 100.0)
    v2_cm = np.abs(df[v2_col].to_numpy(float) * 100.0)

    # both below threshold ⇒ mark as Low Rate and set ratio to NaN
    both_small = (v1_cm < vel_min_cm) & (v2_cm < vel_min_cm)

    # guarded ratio
    denom = np.maximum(v1_cm, vel_min_cm)
    ratio = v2_cm / denom
    ratio[both_small] = np.nan

    # initial grouping from ratio
    group = pd.Series(ratio).apply(categorize_ratio_group).to_numpy(object)

    # overwrite where both are small
    group[both_small] = "Low Rate"

    # write back
    df[f"{label}_v1_cm"] = v1_cm
    df[f"{label}_v2_cm"] = v2_cm
    df[f"ratio_{label}"] = ratio
    df[f"group_{label}"] = group

    return df

# Process all five windows
for v1, v2, lbl, _ttl in PAIRS:
    if v1 not in df.columns or v2 not in df.columns:
        raise ValueError(f"Missing expected columns: {v1} or {v2}")
    df = process_one_period(df, v1, v2, lbl)

# =========================
# Plotting helpers (PyGMT)
# =========================
SCATTER_REGION = [AX_MIN, AX_MAX, AX_MIN, AX_MAX]
SCATTER_PROJ   = "X4.5/4.5c"  # square panels

def plot_one_panel(fig, data, label, CAT_ORDER):
    """Scatter V2 vs V1 for a given 'label' using group colors."""
    # 1:1 reference line
    fig.plot(x=[AX_MIN, AX_MAX], y=[AX_MIN, AX_MAX],
             region=SCATTER_REGION, projection=SCATTER_PROJ,
             pen="0.7p,gray40,4_2:0")

    for g in CAT_ORDER:
        sub = data[data[f"group_{label}"] == g]
        if sub.empty:
            continue
        fig.plot(
            region=SCATTER_REGION, projection=SCATTER_PROJ,
            x=sub[f"{label}_v1_cm"],
            y=sub[f"{label}_v2_cm"],
            style="c0.15c",
            fill=CAT_COLORS[g],
            pen="gray25",
        )
        

# =========================
# Build the figure
# =========================
fig = pygmt.Figure()

# global style tweaks
pygmt.config(
    FONT="10p,Helvetica",
    FONT_TITLE="12p,Helvetica-Bold",
    MAP_FRAME_TYPE="plain",
    MAP_TICK_LENGTH_PRIMARY="2p",
    MAP_TITLE_OFFSET= "4p"
)

with fig.subplot(
    nrows=1, ncols=5,
    figsize=("25c", "4.5c"),
    autolabel="a)",
    margins=["0.3c", "0.2c"],
    sharex="b",   # share x on bottom
    sharey="l",   # share y on left
    frame=["xaf+lV1 (cm/yr, abs)", f"yaf+lV2 (cm/yr, abs)", "WSrt"]):
    
    # Scatter plots
    fig.basemap(region=SCATTER_REGION, projection=SCATTER_PROJ, frame=["WSrt+t0-3 Months", "xa", "ya"], panel=True)
    plot_one_panel(fig, df, "0_3m", CAT_ORDER)
    fig.plot(x = 100, y = 100, style="c0.25c", fill=CAT_COLORS["Low Rate"], pen="gray25", label= "Low Rate+N6")
    fig.plot(x = 100, y = 100, style="c0.25c", fill=CAT_COLORS["Much Slower"], pen="gray25", label= "Much Slower")
    fig.plot(x = 100, y = 100, style="c0.25c", fill=CAT_COLORS["Slower"], pen="gray25", label= "Slower")
    fig.plot(x = 100, y = 100, style="c0.25c", fill=CAT_COLORS["Similar"], pen="gray25", label= "Similar")
    fig.plot(x = 100, y = 100, style="c0.25c", fill=CAT_COLORS["Faster"], pen="gray25", label= "Faster")
    fig.plot(x = 100, y = 100, style="c0.25c", fill=CAT_COLORS["Much Faster"], pen="gray25", label= "Much Faster")
    
    # For multi-column legends users have to provide the width via +w
    with pygmt.config(
            FONT_ANNOT_PRIMARY="11p,black", 
            FONT_ANNOT_SECONDARY="11p,black",
            FONT_LABEL="11p,black",
            ):
        fig.legend(position="JBC+jBC+w20c+o10.5/-1.75c", box=False)
        
    fig.basemap(region=SCATTER_REGION, projection=SCATTER_PROJ, frame=["wSrt+t3-6 Months", "xa", "ya"], panel=True)
    plot_one_panel(fig, df, "3_6m", CAT_ORDER)
    
    fig.basemap(region=SCATTER_REGION, projection=SCATTER_PROJ, frame=["wSrt+t6-9 Months", "xa", "ya"], panel=True)
    plot_one_panel(fig, df, "6_9m", CAT_ORDER)
    
    fig.basemap(region=SCATTER_REGION, projection=SCATTER_PROJ, frame=["wSrt+t9-12 Months", "xa", "ya"], panel=True)
    plot_one_panel(fig, df, "9_12m", CAT_ORDER)
    
    fig.basemap(region=SCATTER_REGION, projection=SCATTER_PROJ, frame=["wSrt+tWater Year", "xa", "ya"], panel=True)
    plot_one_panel(fig, df, "WY", CAT_ORDER)
    

fig.shift_origin(xshift="0c", yshift="-4c")

# We’ll draw 5 rows (one per window), columns = landslides (in df order)
labels_for_rows = ["WY", "9_12m","6_9m", "3_6m", "0_3m"]
n_rows = len(labels_for_rows)
n_cols = len(df)

# Basemap for the heat area
fig.basemap(region=[-20, n_cols, -0.5, n_rows - 0.5],            # y spans row indices centered on integers
    projection="X25c/2c",
    frame=["WSrt", "xa+lLandslide ID (sorted by latitude)"],)

# Put row labels on the left margin (optional)
for r, lab in enumerate(labels_for_rows):
    fig.text(x=-18, y=r, text=lab, font="9p,Helvetica", justify="LM")  # left of the frame
    
# Precompute x positions (column index for each landslide)
x_pos = np.arange(n_cols)

# Category drawing: for each row (time window) and category, plot only the indices in that category
for r, lab in enumerate(labels_for_rows):
    col = f"group_{lab}"
    if col not in df.columns:
        continue

    for g in CAT_ORDER:
        sub_idx = df.index[df[col] == g].to_numpy()  # integer positions in df order
        if sub_idx.size == 0:
            continue

        # y positions are all at the row index r
        y_pos = np.full(sub_idx.size, r)

        fig.plot(
            x=sub_idx, y=y_pos,
            style="y0.4c",               # small square “pixels”
            fill=CAT_COLORS[g],
            pen=None,            # thin gridlines between cells; set to None if you don’t want them
            label=g
        )

fig.basemap(region=[-20, n_cols, -0.5, n_rows - 0.5],            # y spans row indices centered on integers
    projection="X25c/2c",
    frame=["WSrt", "xa+lLandslide ID (sorted by latitude)"],)

# fig.savefig("vel_ratio_panels_pygmt.png", dpi=300)
out_base = f"{fig_dir}/Fig_9_VelRatio_3-6-9-12_minvel{VEL_MIN_CM}_multiple{VEL_MULTIPLE}_intervals"
fig.savefig(out_base + ".png", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(out_base + ".jpeg", crop=True, dpi=300, anti_alias=True, show=False)
fig.savefig(out_base + ".pdf", crop=True, dpi=300, anti_alias=True, show=False)

# Render
fig.show()