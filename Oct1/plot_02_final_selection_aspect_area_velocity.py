#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:45:19 2025

@author: daniellelindsay
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NC_Landslides_paths import *

# -----------------------
# Helpers for robust parsing
# -----------------------
def first_float(x):
    """Return a float from x.
    Handles: float/int; tuple/list/ndarray (takes first element);
    strings like '(0.02,)', '[0.02]', '0.02', 'None', 'nan'.
    Returns np.nan if it cannot parse.
    """
    if x is None:
        return np.nan
    # Already numeric
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    # Sequence
    if isinstance(x, (list, tuple, np.ndarray)):
        return first_float(x[0] if len(x) else np.nan)
    # String-ish
    s = str(x).strip()
    if s.lower() in {"", "none", "nan", "null"}:
        return np.nan
    # Strip outer () or []
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1]
    # Take first item before comma
    if "," in s:
        s = s.split(",", 1)[0].strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def col_to_float_array(series):
    """Vectorized conversion of a pandas Series to a float numpy array using first_float."""
    return series.map(first_float).astype(float).to_numpy()

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(os.path.join(ts_final_dir, "final_selection_only.csv"))
df_all = pd.read_csv(os.path.join(ts_final_dir, "final_selection.csv"))

# Robust parse columns (handles floats/tuples/strings)
aspects_deg      = col_to_float_array(df['ls_mean_aspect'])
areas            = col_to_float_array(df['ls_area_m2'])
velocities_cmyr  = col_to_float_array(df['ts_linear_vel_myr']) * 100.0  # cm/yr

aspects_deg_all  = col_to_float_array(df_all['ls_mean_aspect'])
areas_all        = col_to_float_array(df_all['ls_area_m2'])

# -----------------------
# Filtering masks
# -----------------------
# Keep finite aspects/areas/velocities and positive areas for log radius
mask = np.isfinite(aspects_deg) & np.isfinite(areas) & np.isfinite(velocities_cmyr) & (areas > 0)
aspects_deg = aspects_deg[mask]
areas = areas[mask]
velocities_cmyr = velocities_cmyr[mask]

mask_all = np.isfinite(aspects_deg_all) & np.isfinite(areas_all) & (areas_all > 0)
aspects_deg_all = aspects_deg_all[mask_all]
areas_all = areas_all[mask_all]

# Guard against empty after filtering
if aspects_deg.size == 0 or areas.size == 0:
    raise RuntimeError("No valid (finite, positive-area) rows in final_selection_only.csv to plot.")
if aspects_deg_all.size == 0 or areas_all.size == 0:
    print("⚠️ No valid rows in final_selection.csv after filtering; plotting only selected set as colored points.")
    plot_all = False
else:
    plot_all = True

# Convert aspects to radians
angles = np.deg2rad(aspects_deg)
angles_all = np.deg2rad(aspects_deg_all) if plot_all else np.array([])

# Sort by absolute velocity so largest magnitudes plot last
order = np.argsort(np.abs(velocities_cmyr))
angles_sorted = angles[order]
areas_sorted = areas[order]
velocities_sorted = velocities_cmyr[order]

# Sort the "all" set by area magnitude (so largest areas are drawn last)
if plot_all:
    order_all = np.argsort(np.abs(areas_all))
    angles_sorted_all = angles_all[order_all]
    areas_sorted_all = areas_all[order_all]

# Satellite look direction (φ = bearing CW from north)
phi = np.deg2rad(90 + 11)  # ALOS-2 right-looking ascending (example)

# Background sensitivity shading grid
az_grid = np.linspace(0, 2*np.pi, 360)
S_norm = np.abs(np.cos(az_grid - phi))  # normalized horizontal sensitivity
brightness = 0.5 + 0.5 * S_norm
bg_colors = plt.cm.gray(brightness)

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rscale('log')
ax.set_axisbelow(True)

# Determine radial extent safely
r_min = max(np.nanmin(areas) * 0.9, 1e-6)  # avoid zero/negative on log scale
r_max = np.nanmax(areas) * 1.1

# Draw shading as wedges
width = 2*np.pi / len(az_grid)
ax.bar(az_grid, np.full_like(az_grid, r_max), width=width, bottom=0,
       color=bg_colors, edgecolor='none', zorder=0)




# Grid and styling
ax.grid(True, zorder=2)

# Scatter: all landslides (grey) for context
if plot_all:
    ax.scatter(
        angles_sorted_all,
        areas_sorted_all,
        c="gray",
        s=30,
        alpha=0.5,
        edgecolor='k',
        linewidth=0.5,
        zorder=1
    )

# Scatter: selected landslides colored by LOS velocity (cm/yr)
sc = ax.scatter(
    angles_sorted,
    areas_sorted,
    c=velocities_sorted,
    cmap='RdYlBu_r',
    vmin=-10,
    vmax=10,
    s=30,
    alpha=1,
    edgecolor='k',
    linewidth=0.5,
    zorder=3
)

# Colorbar on right
cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.1, shrink=0.55)
cbar.set_label('LOS Velocity (cm/yr)')

# Title above plot
ax.set_title('Landslide Aspect vs. Area (log)', y=1.1)

# Radial limits
ax.set_ylim(r_min, r_max)

# Save outputs
jpg_path = f'{fig_dir}/Figure_2_landslide_aspect_area_velocity_log.jpg'
png_path = f'{fig_dir}/Figure_2_landslide_aspect_area_velocity_log.png'
pdf_path = f'{fig_dir}/Figure_2_landslide_aspect_area_velocity_log.pdf'
os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
fig.savefig(jpg_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')

plt.tight_layout()
plt.show()
