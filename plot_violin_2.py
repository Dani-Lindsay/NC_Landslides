#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:44:05 2025

@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv")

# Compute absolute velocities in cm/yr
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)

# Minimum velocity threshold to define significant landslides
vel_min_threshold = 2
df = df[(df["vel_dry1"] > vel_min_threshold) | (df["vel_dry2"] > vel_min_threshold)].copy()

# Compute velocity ratio
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

# Define velocity change categories
def categorize_ratio_group(ratio):
    if ratio < 0.333:
        return "Much slower in WY23"
    elif 0.333 <= ratio < 0.83:
        return "Slower in WY23"
    elif 0.83 <= ratio <= 1.2:
        return "Similar speed"
    elif 1.2 < ratio <= 3.0:
        return "Faster in WY23"
    elif ratio > 3.0:
        return "Much faster in WY23"
    else:
        return "Low Rate"

df["group"] = df["vel_ratio"].apply(categorize_ratio_group)

# Define variables for violin plots
violin_vars = [
    "support_params/wy23_vs_wy22_rain_ratio",
    "support_params/wy23_vs_wy22_pga_ratio",
    "support_params/wy23_vs_wy22_eq_ratio",
    "meta__ls_area_m2",
    "meta__ls_mean_slope",
    "meta__ls_axis_ratio"
]

# Apply log10 to area to avoid skew
df["meta__ls_area_m2_log10"] = np.log10(df["meta__ls_area_m2"])
violin_labels = {
    "support_params/wy23_vs_wy22_rain_ratio": "Rain Ratio (WY23/WY22)",
    "support_params/wy23_vs_wy22_pga_ratio": "PGA Ratio (WY23/WY22)",
    "support_params/wy23_vs_wy22_eq_ratio": "EQ Count Ratio (WY23/WY22)",
    "meta__ls_area_m2_log10": "Log Area (m²)",
    "meta__ls_mean_slope": "Mean Slope (°)",
    "meta__ls_axis_ratio": "Axis Ratio"
}

# Replace area variable with log10 version for plotting
plot_vars = violin_vars.copy()
plot_vars[violin_vars.index("meta__ls_area_m2")] = "meta__ls_area_m2_log10"

# Set up the plot grid
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

# Plot each violin plot
for i, var in enumerate(plot_vars):
    sns.violinplot(data=df, x="group", y=var, ax=axes[i], inner="quart", gap=.1)
    axes[i].set_title(violin_labels[var], fontsize=10)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(labelsize=8)

# Overall layout
fig.suptitle("Comparison of Landslide Groups by External and Geometric Variables", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/mnt/data/velocity_response_violin_grid.png", dpi=300)
plt.show()
