#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 07:11:31 2025

@author: daniellelindsay
"""

# Sign-mismatch analysis + single multi-panel figure
# - Reads final_selection.csv
# - Computes expected sign from aspect (10–190° => +1; else -1)
# - Finds mismatches (observed != expected)
# - Prints the stats needed for your paragraph
# - Saves ONE figure with subplots (boxplots + boundary-distance histogram)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --------- EDIT IF NEEDED ----------
CSV_PATH = r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv"
ASPECT_COL   = "ls_mean_aspect"          # degrees
SIGN_COL     = "ls_sign"                  # observed sign (+1/-1)
VEL_COL      = "ts_linear_vel_myr"        # LOS velocity (m/yr)
BGSTD_COL    = "ts_background_std_my-1"   # background std (m/yr)
CLUSTER_COL  = "ts_cluster_area_m2"       # cluster area (m^2)
NN_COL       = "ts_mean_nn_dist_m"        # mean NN distance (m)
SELECTED_COL = "selected"                 # analyze selected==True as the active set
LOW_DEG, HIGH_DEG = 10.0, 190.0           # aspect bounds for expected + sign
OUT_FIG = r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/sign_mismatch_panels.png"
# -----------------------------------

def to_bool(s):
    return s.astype(str).str.lower().isin(["true","t","1","yes","y","on"])

def series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

def abs_series(df, col):
    return series(df, col).abs()

def expected_sign_and_aspect(df, col, lo=LOW_DEG, hi=HIGH_DEG):
    a = (pd.to_numeric(df[col], errors="coerce") % 360.0)
    exp = np.where((a >= lo) & (a <= hi), 1, -1)
    return pd.Series(exp, index=df.index), a

def dist_to_boundary(a_deg, lo=LOW_DEG, hi=HIGH_DEG):
    if pd.isna(a_deg): return np.nan
    a = float(a_deg) % 360.0
    return min(abs(a-lo), abs(a-hi))

# --- Load + filter to active set (selected==True) ---
df_all = pd.read_csv(CSV_PATH)
df = df_all[to_bool(df_all[SELECTED_COL])] if SELECTED_COL in df_all.columns else df_all.copy()

# --- Compute expected vs observed sign and mismatches ---
exp_sign, aspect_norm = expected_sign_and_aspect(df, ASPECT_COL)
obs_sign = pd.to_numeric(df[SIGN_COL], errors="coerce")

valid_mask = (~aspect_norm.isna()) & (~obs_sign.isna())
n_valid = int(valid_mask.sum())

mismatch_mask = valid_mask & (obs_sign.astype(int) != exp_sign.astype(int))
mismatches = df[mismatch_mask].copy()
mismatches["aspect_norm_deg"] = aspect_norm[mismatch_mask]
mismatches["expected_sign"] = exp_sign[mismatch_mask].astype(int)
mismatches["observed_sign"] = obs_sign[mismatch_mask].astype(int)
mismatches["dist_to_boundary_deg"] = mismatches["aspect_norm_deg"].apply(dist_to_boundary)

# "others" = active set minus mismatches
others = df.loc[valid_mask & ~mismatch_mask].copy()

# --- Build series for stats/plots ---
v_mism, v_other   = abs_series(mismatches, VEL_COL), abs_series(others, VEL_COL)
bg_mism, bg_other = series(mismatches, BGSTD_COL), series(others, BGSTD_COL)
ca_mism, ca_other = series(mismatches, CLUSTER_COL), series(others, CLUSTER_COL)
nn_mism, nn_other = series(mismatches, NN_COL), series(others, NN_COL)
dists = pd.to_numeric(mismatches["dist_to_boundary_deg"], errors="coerce").dropna()

# --- Stats for your sentence ---
n_mismatch = len(mismatches)
pct_mismatch = 100.0 * n_mismatch / n_valid if n_valid > 0 else np.nan
within10 = int((dists <= 10).sum())
pct_within10 = 100.0 * within10 / len(dists) if len(dists) > 0 else np.nan

def med_str(s):  # median with fallback
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.median(s)) if len(s) else np.nan

med_v_mism, med_v_other = med_str(v_mism), med_str(v_other)
med_bg_mism, med_bg_other = med_str(bg_mism), med_str(bg_other)
med_ca_mism, med_ca_other = med_str(ca_mism), med_str(ca_other)

print("=== Stats for manuscript sentence ===")
print(f"Mismatches: {n_mismatch}/{n_valid} ({pct_mismatch:.1f}%)")
print(f"Within 10° of 10°/190°: {within10}/{len(dists)} ({pct_within10:.1f}%)")
print(f"|Velocity| median (m/yr): mismatches={med_v_mism:.3g}, others={med_v_other:.3g}")
print(f"Background std median (m/yr): mismatches={med_bg_mism:.3g}, others={med_bg_other:.3g}")
print(f"Cluster area median (m^2): mismatches={med_ca_mism:.3g}, others={med_ca_other:.3g}")

# --- Single figure with subplots (4 boxplots + 1 histogram) ---
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1,1,1])

# (0,0) Background std
ax1 = fig.add_subplot(gs[0,0])
ax1.boxplot([bg_other.dropna(), bg_mism.dropna()], labels=["others","mismatches"], whis=1.5, showfliers=False)
ax1.set_title("Background variability")
ax1.set_ylabel("background std (m/yr)")

# (0,1) Absolute velocity
ax2 = fig.add_subplot(gs[0,1])
ax2.boxplot([v_other.dropna(), v_mism.dropna()], labels=["others","mismatches"], whis=1.5, showfliers=False)
ax2.set_title("Absolute velocity")
ax2.set_ylabel("|velocity| (m/yr)")

# (1,0) Cluster area
ax3 = fig.add_subplot(gs[1,0])
ax3.boxplot([ca_other.dropna(), ca_mism.dropna()], labels=["others","mismatches"], whis=1.5, showfliers=False)
ax3.set_title("Kinematic cluster area")
ax3.set_ylabel("cluster area (m^2)")

# (1,1) Mean NN distance (if available)
ax4 = fig.add_subplot(gs[1,1])
if len(nn_other.dropna()) and len(nn_mism.dropna()):
    ax4.boxplot([nn_other.dropna(), nn_mism.dropna()], labels=["others","mismatches"], whis=1.5, showfliers=False)
    ax4.set_ylabel("mean NN distance (m)")
else:
    ax4.text(0.5, 0.5, "NN distance not available", ha="center", va="center")
    ax4.set_xticks([]); ax4.set_yticks([])
ax4.set_title("Mean nearest-neighbor dist.")

# (2,0:2) Histogram of boundary-distance for mismatches
ax5 = fig.add_subplot(gs[2, :])
if len(dists):
    ax5.hist(dists, bins=np.arange(0, 91, 5))
    ax5.set_xlabel("distance to 10° or 190° (deg)")
    ax5.set_ylabel("count")
    ax5.set_title("Distance to sign-change boundaries (mismatches only)")
    ax5.axvline(5, linestyle="--")
    ax5.axvline(10, linestyle="--")
else:
    ax5.text(0.5, 0.5, "No aspects available for boundary distance.", ha="center", va="center")
    ax5.set_axis_off()

fig.suptitle("Sign-mismatch evidence", y=0.98)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print(f"\nSaved figure: {OUT_FIG}")

# ---- Example sentence you can paste (printed for convenience) ----
print("\nSuggested sentence:")
# print(
#     f\"We set the landslide sign from the line-of-sight cumulative displacement and find only {n_mismatch}/{n_valid} "
#     f"({pct_mismatch:.1f}\\%) with the opposite sign to that expected from aspect; about {within10}/{len(dists)} "
#     f"({pct_within10:.1f}\\%) of these lie within 10$^\\circ$ of the 10$^\\circ$/190$^\\circ$ boundaries. "
#     f"Relative to the rest of the active set, they show comparable |velocity| (median {med_v_mism:.3g} vs. {med_v_other:.3g} m/yr), "
#     f"lower background variability (median {med_bg_mism:.3g} vs. {med_bg_other:.3g} m/yr), and smaller kinematic clusters "
#     f"(median {med_ca_mism:.3g} vs. {med_ca_other:.3g} m^2).\n")
    

print(f"We set the landslide sign from the line-of-sight cumulative displacement and find only "
    f"{n_mismatch}/{n_valid} ({pct_mismatch:.1f}\\%) with the opposite sign to that expected from aspect; "
    f"about")

