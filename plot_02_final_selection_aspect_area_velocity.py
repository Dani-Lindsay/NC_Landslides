#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:45:19 2025

@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NC_Landslides_paths import *

# Load data
df = pd.read_csv('/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection_only.csv')

aspects_deg = df['ls_mean_aspect'].values
areas = df['ls_area_m2'].values
velocities = df['ts_linear_vel_myr'].values*100

# Filter out non-positive areas for log scale
mask = areas > 0
aspects_deg = aspects_deg[mask]
areas = areas[mask]
velocities = velocities[mask]

# Convert aspects to radians
angles = np.deg2rad(aspects_deg)

# Sort by absolute velocity so largest magnitudes plot last
order = np.argsort(np.abs(velocities))
angles_sorted = angles[order]
areas_sorted = areas[order]
velocities_sorted = velocities[order]

# Satellite look direction (φ = bearing CW from north)
phi = np.deg2rad(90+11)  # ALOS-2 right-looking ascending

# Background sensitivity shading grid
az_grid = np.linspace(0, 2*np.pi, 360)
S_norm = np.abs(np.cos(az_grid - phi))  # normalized horizontal sensitivity
brightness = 0.5 + 0.5 * S_norm
bg_colors = plt.cm.gray(brightness)

# Plot
fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection':'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rscale('log')
ax.set_axisbelow(True)

# Draw shading as wedges
max_r = areas.max() * 1.1
width = 2*np.pi / len(az_grid)
ax.bar(az_grid, np.full_like(az_grid, max_r), width=width, bottom=0,
       color=bg_colors, edgecolor='none', zorder=0,)

# Grid and styling
ax.grid(True, zorder=1)

# Scatter: overlay landslides colored by velocity
sc = ax.scatter(
    angles_sorted,
    areas_sorted,
    c=velocities_sorted,
    cmap='RdYlBu_r',
    vmin=-10,
    vmax=10,
    s=45,
    alpha=0.9,
    edgecolor='k',
    linewidth=0.5,
    zorder=2
)

# Colorbar on right
cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.1, shrink=0.55)
cbar.set_label('LOS Velocity (cm/yr)')

# Title above plot
ax.set_title('Landslide Aspect vs. Area (log)',
             y=1.08)

# Radial limits
ax.set_ylim(areas.min()*0.9, areas.max()*1.1)

# Save outputs
jpg_path = f'{fig_dir}/landslide_aspect_area_velocity_log.jpg'
png_path = f'{fig_dir}/landslide_aspect_area_velocity_log.png'
pdf_path = f'{fig_dir}/landslide_aspect_area_velocity_log.pdf'
fig.savefig(jpg_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')

plt.tight_layout()
plt.show()
