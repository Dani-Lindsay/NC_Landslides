#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 21:58:47 2025

@author: daniellelindsay
"""

# Count how many selected landslides are within 500 m of roads/coast (HDF5 per-ID)
# Edit paths below. Requires: pandas, numpy, h5py.

import os
import re
import numpy as np
import pandas as pd
import h5py

# ----------------- EDIT THESE -----------------
CSV_PATH   = r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv"      # your inventory CSV
H5_DIR     = r"/Volumes/Seagate/NC_Landslides/Inputs/landslide_supporting"                # folder containing per-ID HDF5 files
H5_TEMPLATE = "{ls_id}-supporting.h5"             # how to build the filename from ls_id
ID_COL     = "ls_id"                              # ID column in CSV (e.g., "ls_id")
SELECTED_COL = "selected"                         # boolean/boolean-like (True/False) column
DIST_THRESH_M = 500.0                             # threshold in meters
DIST_THRESH_M_coast = 1000
# ----------------------------------------------

TRUE_SET  = {"true","t","1","yes","y","on"}
FALSE_SET = {"false","f","0","no","n","off"}

def to_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in TRUE_SET:  return True
    if s in FALSE_SET: return False
    try:
        v = float(s)
        if v == 1: return True
        if v == 0: return False
    except Exception:
        pass
    return np.nan

def read_scalar_or_min(dset):
    """Return a float from an HDF5 dataset that might be scalar or array."""
    arr = np.array(dset)
    if arr.size == 0:
        return np.nan
    try:
        return float(np.nanmin(arr.astype(float)))
    except Exception:
        # last resort: try direct cast
        try:
            return float(arr)
        except Exception:
            return np.nan

def check_distances_for_id(h5_path, group_name):
    """
    For a given file and group (e.g., 'ls_001'), read:
      /{group}/payload/distances/road_m
      /{group}/payload/distances/ocean_m
    Returns (road_m, ocean_m) as floats (nan if missing).
    """
    road_m = np.nan
    ocean_m = np.nan
    if not os.path.exists(h5_path):
        return road_m, ocean_m

    with h5py.File(h5_path, "r") as f:
        road_key  = f"/{group_name}/payload/distances/road_m"
        coast_key = f"/{group_name}/payload/distances/ocean_m"
        if road_key in f:
            road_m = read_scalar_or_min(f[road_key])
        if coast_key in f:
            ocean_m = read_scalar_or_min(f[coast_key])
    return road_m, ocean_m

# ---- Run ----
df = pd.read_csv(CSV_PATH)
sel_bool = df[SELECTED_COL].apply(to_bool)
df_sel = df[sel_bool == True].copy()

results = []
missing_files = 0

for _, row in df_sel.iterrows():
    ls_id = str(row[ID_COL])
    # filename from template
    h5_file = H5_TEMPLATE.format(ls_id=ls_id)
    h5_path = os.path.join(H5_DIR, h5_file)

    # the H5 group usually matches the ID (e.g., 'ls_001')
    group_name = ls_id

    road_m, ocean_m = check_distances_for_id(h5_path, group_name)
    file_exists = os.path.exists(h5_path)
    if not file_exists:
        missing_files += 1

    within_road  = (road_m  <= DIST_THRESH_M) if np.isfinite(road_m) else False
    within_coast = (ocean_m <= DIST_THRESH_M_coast) if np.isfinite(ocean_m) else False

    results.append({
        "ls_id": ls_id,
        "h5_file": h5_file,
        "h5_found": file_exists,
        "road_m": road_m,
        "ocean_m": ocean_m,
        f"within_{int(DIST_THRESH_M)}m_road": within_road,
        f"within_{int(DIST_THRESH_M_coast)}m_coast": within_coast,
        "within_both": (within_road and within_coast),
    })

res_df = pd.DataFrame(results)

# Summary counts (only among selected==True)
n_selected = len(df_sel)
n_with_files = int(res_df["h5_found"].sum()) if not res_df.empty else 0
n_within_road = int(res_df[f"within_{int(DIST_THRESH_M)}m_road"].sum()) if not res_df.empty else 0
n_within_coast = int(res_df[f"within_{int(DIST_THRESH_M_coast)}m_coast"].sum()) if not res_df.empty else 0
n_within_both = int(res_df["within_both"].sum()) if not res_df.empty else 0

def pct(x, n): 
    return "N/A" if n == 0 else f"{100.0 * x / n:.1f}%"

print("\n=== Proximity to Road/Coast for SELECTED landslides ===\n")
print(f"Selected (True):            {n_selected}")
print(f"H5 found:                   {n_with_files} ({pct(n_with_files, n_selected)})")
print(f"Within {int(DIST_THRESH_M)} m of a road:  {n_within_road} ({pct(n_within_road, n_selected)})")
print(f"Within {int(DIST_THRESH_M_coast)} m of the coast: {n_within_coast} ({pct(n_within_coast, n_selected)})")
print(f"Within both:                {n_within_both} ({pct(n_within_both, n_selected)})")

# Optional: save per-ID results
out_csv = os.path.splitext(CSV_PATH)[0] + f"_selected_proximity_{int(DIST_THRESH_M_coast)}m_coast.csv"
res_df.to_csv(out_csv, index=False)
print(f"\nSaved per-ID results: {out_csv}")

# --- Config (tweak if needed) ---
CSV_PATH = CSV_PATH if 'CSV_PATH' in globals() else "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv"
SLOPE_COL = "ls_mean_slope"   # column with mean slope
DIST_COAST_COL = "dist_coast_m"  # distance to coast in meters (if absent, set to None and use your own mask)
COAST_THRESH_M = 500.0        # coastal threshold in meters (match your study)

import pandas as pd
import numpy as np
import os

# --- Load data if df not already present ---
if 'df' not in globals():
    df = pd.read_csv(CSV_PATH)

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _slope_to_degrees(s: pd.Series) -> pd.Series:
    """
    Convert slope to degrees if needed:
    - If values look like radians (max <= ~1.6), convert rad->deg.
    - If values look normalized (0..1.0-ish, max <= 1.0), map to 0..90° by ×90.
    - Otherwise assume already in degrees.
    """
    x = _coerce_numeric(s).astype(float)
    if x.dropna().empty:
        return x
    smax = np.nanmax(x)
    if smax <= 1.6 and np.nanmean(x) < 1.0:   # likely radians
        return np.degrees(x)
    if smax <= 1.0:                           # likely normalized 0..1
        return x * 90.0
    return x                                   # already degrees

def slope_bins_percent(s_deg: pd.Series):
    """Return ordered dict of (count, pct) for bins <20, 20–25, 25–35, 35–45, >45°."""
    s = _coerce_numeric(s_deg).dropna()
    if s.empty:
        return {}
    n = len(s)
    bins = {
        "<20°":      (s < 20).sum(),
        "20–25°":    ((s >= 20) & (s <= 25)).sum(),
        "25–35°":    ((s >= 25) & (s <= 35)).sum(),
        "35–45°":    ((s >= 35) & (s <= 45)).sum(),
        ">45°":      (s > 45).sum(),
    }
    return {k: (v, 100.0*v/n) for k, v in bins.items()}

# --- Build masks: coastal vs inland ---
if DIST_COAST_COL in df.columns:
    dist_coast = _coerce_numeric(df[DIST_COAST_COL])
    coastal_mask = dist_coast <= COAST_THRESH_M
else:
    # If you don't have a distance column in the CSV, you can import a coastal ID list and build a mask instead.
    raise KeyError(f"'{DIST_COAST_COL}' not found in df. Provide a coastal mask or add the distance column.")

inland_mask = ~coastal_mask

# --- Prepare slope in degrees ---
if SLOPE_COL not in df.columns:
    raise KeyError(f"'{SLOPE_COL}' not found in df.")

s_all_deg = _slope_to_degrees(df[SLOPE_COL])
s_coast = s_all_deg[coastal_mask]
s_inland = s_all_deg[inland_mask]

# --- Stats ---
med_coast = float(np.nanmedian(s_coast)) if s_coast.notna().any() else np.nan
med_inland = float(np.nanmedian(s_inland)) if s_inland.notna().any() else np.nan

bins_inland = slope_bins_percent(s_inland)
bins_coast  = slope_bins_percent(s_coast)

def fmt_bins_line(bins):
    # "<20°: 86 (19.7%); 20–25°: 85 (19.5%); ..."
    if not bins: return "N/A"
    return "; ".join([f"{k}: {v[0]} ({v[1]:.1f}%)" for k, v in bins.items()])

# --- Console summary ---
print("\n=== Slope metrics by region ===")
print(f"Coastal median slope: {med_coast:.1f}°  |  Inland median slope: {med_inland:.1f}°")
print(f"Inland slope bins: {fmt_bins_line(bins_inland)}")
print(f"Coastal slope bins: {fmt_bins_line(bins_coast)}")

# --- LaTeX-ready lines for manuscript ---
latex_coast = (
    rf"Coastal landslides are much steeper, with a median slope of {med_coast:.1f}$^\circ$ "
    r"(Figure S\ref{fig:map_aspect_slope_height})."
)

# Expand inland bin percentages into the requested narrative order
def pct(b, key): 
    return f"{b[key][1]:.1f}\\%" if key in b else "N/A"
inland_line = (
    r"For the inland slides, we find "
    rf"{pct(bins_inland,'<20°')} of slopes are shallow ($\leq$ 20$^\circ$), "
    rf"{pct(bins_inland,'20–25°')} are moderate (20–25$^\circ$), "
    rf"{pct(bins_inland,'25–35°')} are steep (25–35$^\circ$), "
    rf"{pct(bins_inland,'35–45°')} are very steep (35–45$^\circ$), and "
    rf"{pct(bins_inland,'>45°')} are extremely steep ($\geq$ 45$^\circ$) "
    r"(Figure \ref{fig:inventory_variables})."
)

print("\n--- LaTeX copy/paste ---")
print(latex_coast)
print(inland_line)

