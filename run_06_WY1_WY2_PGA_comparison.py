#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity vs PGA grouping analysis (Option 2 thresholds).
Groups landslides by PGA ratio, plots velocities colored by PGA bins.
"""

# =========================
# Imports & config
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from NC_Landslides_paths import *  # provides fig_dir etc.

# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 2  # cm/yr (to filter out nearly-stable slides)

# =========================
# Load & prepare data
# =========================
df = pd.read_csv(
    "/Volumes/Seagate/NC_Landslides/Data_1/LS_Final_TS_4/compiled_landslide_data.csv"
)

# Absolute velocities in cm/yr
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

# PGA ratio (unitless)
df["pga_ratio"] = df["support_params/wy23_vs_wy22_pga_ratio"]

# Filter out very low-velocity slides if you want
df = df[(df["vel_dry1"] > vel_min_threshold) | (df["vel_dry2"] > vel_min_threshold)].copy()

# -------------------------
# Grouping function (Option 2 thresholds)
# -------------------------
def categorize_pga_group(ratio):
    if ratio < 0.6:
        return "Much Lower"
    elif 0.6 <= ratio < 0.83:
        return "Lower"
    elif 0.83 <= ratio <= 1.2:
        return "Similar"
    elif 1.2 < ratio <= 3:
        return "Higher"
    else:
        return "Much Higher"

df["group"] = df["pga_ratio"].apply(categorize_pga_group)

# -------------------------
# Colors: use diverging palette
# -------------------------
group_order = ["Much Lower", "Lower", "Similar", "Higher", "Much Higher"]

# Get diverging palette (Spectral) and reverse it
diverging_colors = sns.color_palette("Spectral", n_colors=len(group_order))[::-1]

# Map to groups
roma_colors = dict(zip(group_order, diverging_colors))
# =========================
# Scatter plot (vel_dry1 vs vel_dry2, colored by PGA group)
# =========================
fig, ax = plt.subplots(figsize=(5, 5))
max_val = 20
ax.plot([0, max_val], [0, max_val], linestyle="--", color="black", linewidth=1, alpha=0.5)


group_order = ["Much Lower", "Lower", "Similar", "Higher", "Much Higher"]


for g in group_order:
    sub = df[df["group"] == g]
    ax.scatter(
        sub["vel_dry1"], sub["vel_dry2"],
        label=g, s=30, color=roma_colors[g],
        edgecolor="black", linewidth=0.3
    )

ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_box_aspect(1)

ax.set_xlabel("Dry Season Velocity WY22 (cm/yr, abs)", fontsize=10)
ax.set_ylabel("Dry Season Velocity WY23 (cm/yr, abs)", fontsize=10)
ax.set_title("Velocity Comparison by PGA Ratio Group", fontsize=10)
ax.tick_params(labelsize=10)

# Custom legend in group order
ax.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=roma_colors[g],
                        markeredgecolor="black", markersize=6, label=g)
             for g in group_order],
    title="PGA Ratio Group", fontsize=9, title_fontsize=10,
    loc="upper right"
)

plt.tight_layout()
plt.savefig(f"{fig_dir}/Fig_Velocity_scatter_byPGA_opt2.png", dpi=300)
plt.savefig(f"{fig_dir}/Fig_Velocity_scatter_byPGA_opt2.jpeg", dpi=300)
plt.savefig(f"{fig_dir}/Fig_Velocity_scatter_byPGA_opt2.pdf")
plt.show()

# Drop extreme outliers in velocity ratio
# Example: anything above the 99th percentile
q_hi = df["vel_ratio"].quantile(0.99)
df_clip = df[df["vel_ratio"] <= q_hi].copy()

print(f"Dropped {(len(df) - len(df_clip))} extreme outliers above {q_hi:.2f}")

# =========================
# Violin plot (velocity ratio by PGA group, clipped)
# =========================
sns.set(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(
    data=df_clip, x="group", y="vel_ratio",
    order=group_order, palette=roma_colors,
    linewidth=1, inner="quart", scale="width", ax=ax
)

ax.set_xlabel("PGA Ratio Group", fontsize=10)
ax.set_ylabel("Velocity Ratio (WY23/WY22)", fontsize=10)
ax.set_title("Velocity Ratio Distribution by PGA Group (clipped at 99th percentile)", fontsize=10)
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
fig.savefig(f"{fig_dir}/Fig_VelocityRatio_violin_byPGA_opt2_clipped.png", dpi=300)
fig.savefig(f"{fig_dir}/Fig_VelocityRatio_violin_byPGA_opt2_clipped.jpeg", dpi=300)
fig.savefig(f"{fig_dir}/Fig_VelocityRatio_violin_byPGA_opt2_clipped.pdf")
plt.show()


# -------------------------
# Proportion of accelerations per group
# -------------------------

# Define acceleration as vel_ratio > 1 (WY23 faster than WY22)
accel_stats = (
    df.groupby("group")["vel_ratio"]
      .apply(lambda x: (x > 1).mean())  # proportion accelerating
      .rename("prop_accelerated")
)

# Also count how many landslides in each group
counts = df["group"].value_counts().reindex(group_order, fill_value=0)

# Combine into summary table
summary = pd.concat([counts.rename("count"), accel_stats], axis=1)

print("\nAcceleration summary by group:")
print(summary.to_string(float_format="%.2f"))

