#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:26:55 2025

@author: daniellelindsay
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from NC_Landslides_paths import *

# --- seaborn style ---
sns.set_theme(style="white", context="talk")

# Load the data
df = pd.read_csv(os.path.join(ts_final_dir, "final_selection_only.csv"))

# Build clean variables (readable units & scales)
df_plot = pd.DataFrame({
    # angles already in degrees
    "Aspect (deg)": df["ls_mean_aspect"],
    "Slope (deg)": df["ls_mean_slope"]*90,
    # convert m/yr -> cm/yr
    "LOS velocity (cm/yr)": df["ts_linear_vel_myr"] * 100.0,
    # log10 of area for readability across orders of magnitude
    "Area (m², log10)": np.log10(df["ls_area_m2"]),
    "Axis ratio": df["ls_axis_ratio_x"],
})

# Remove non-finite rows
df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()

# Pairplot
g = sns.pairplot(
    df_plot,
    corner=True,                 # lower triangle only (cleaner)
    diag_kind="hist",
    plot_kws=dict(s=18, alpha=0.8, edgecolor="white", linewidth=0.3),
    diag_kws=dict(bins=20, edgecolor="white"),
)

# Nice suptitle and tight layout
plt.suptitle("Landslide Inventory Variables", y=1.02)
plt.tight_layout()

# Save
out_base = f"{fig_dir}/Figure_3_Landslide_Inventory_Scatter"
plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
plt.savefig(out_base + ".jpeg", dpi=300, bbox_inches="tight")
plt.savefig(out_base + ".pdf", bbox_inches="tight")
plt.show()
