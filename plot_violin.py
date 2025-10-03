#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:37:32 2025

@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Volumes/Seagate/NC_Landslides/Data_1/LS_Final_TS_4/compiled_landslide_data.csv")


# Load and preprocess (same as your script)
#df = pd.read_csv("compiled_landslide_data.csv")
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
df = df[(df["vel_dry1"] > 2) | (df["vel_dry2"] > 2)].copy()
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

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

# Set consistent group order
group_order = [
    "Much slower in WY23",
    "Slower in WY23",
    "Similar speed",
    "Faster in WY23",
    "Much faster in WY23"
]

# Set variables to plot
variables = {
    "support_params/wy23_vs_wy22_rain_ratio": "Rainfall Ratio (WY23/WY22)",
    "support_params/wy23_vs_wy22_pga_ratio": "PGA Ratio (WY23/WY22)",
    "support_params/wy23_vs_wy22_eq_ratio": "EQ Count Ratio (WY23/WY22)",
    "support_params/dist_ocean_m": "Distance from Ocean (m)",
    "support_params/area_m2": "Landslide Area (m²)",
}

# Plot
fig, axes = plt.subplots(nrows=1, ncols=len(variables), figsize=(5 * len(variables), 5), sharey=False)

for ax, (var, label) in zip(axes, variables.items()):
    sns.violinplot(data=df, x="group", y=var, order=group_order,
                   ax=ax, palette="Spectral", gap=0.1, inner="quart", linewidth=0.7)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=45)

fig.suptitle("Environmental Factors by Velocity Change Group", fontsize=14)
plt.tight_layout()
plt.show()
