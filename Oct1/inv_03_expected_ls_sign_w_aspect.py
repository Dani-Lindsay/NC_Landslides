#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 06:09:54 2025

@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import os
from NC_Landslides_paths import *

# --- Edit if needed ---
CSV_PATH = os.path.join(ts_final_dir, "final_selection_only.csv")
ASPECT_COL = "ls_mean_aspect"   # degrees
SIGN_COL   = "ls_sign"          # observed sign (+1 / -1)
# If you want to analyze only selected==True, set this to True
SELECTED_ONLY = False
SELECTED_COL  = "selected"
# ----------------------

df = pd.read_csv(CSV_PATH)

# Optional filter
if SELECTED_ONLY and (SELECTED_COL in df.columns):
    sel = df[SELECTED_COL].astype(str).str.lower().isin(["true","t","1","yes","y","on"])
    df = df[sel].copy()

# Normalize aspects to [0, 360)
aspect = pd.to_numeric(df[ASPECT_COL], errors="coerce") % 360

# Expected sign: +1 if aspect in [10, 190], else -1
expected_sign = np.where((aspect >= 10) & (aspect <= 190), 1, -1)

# Observed sign
obs_sign = pd.to_numeric(df[SIGN_COL], errors="coerce")

valid_mask = (~obs_sign.isna()) & (~pd.isna(aspect))
obs = obs_sign[valid_mask].astype(int)
exp = pd.Series(expected_sign, index=df.index)[valid_mask].astype(int)

# Build mismatch mask
mismatch_mask = valid_mask & (obs != exp)

# Create a DataFrame of mismatches with helpful columns
mismatches_df = df.loc[mismatch_mask].copy()
mismatches_df["aspect_norm_deg"] = aspect.loc[mismatch_mask]
mismatches_df["expected_sign"]   = exp.loc[mismatch_mask]
mismatches_df["observed_sign"]   = obs.loc[mismatch_mask]

# Direction label
def label_direction(row):
    if row["expected_sign"] == -1 and row["observed_sign"] == 1:
        return "expected_neg_observed_pos"
    if row["expected_sign"] == 1 and row["observed_sign"] == -1:
        return "expected_pos_observed_neg"
    return "other"

mismatches_df["mismatch_direction"] = mismatches_df.apply(label_direction, axis=1)

# (Optional) counts by direction
counts = mismatches_df["mismatch_direction"].value_counts().to_dict()

# Save to CSV next to the input file
out_csv = os.path.splitext(CSV_PATH)[0] + "_sign_mismatches.csv"
mismatches_df.to_csv(out_csv, index=False)

# Print a quick summary
n_valid = int(valid_mask.sum())
n_mismatch = int(mismatch_mask.sum())
pct_mismatch = (100.0 * n_mismatch / n_valid) if n_valid > 0 else np.nan

print("=== Opposite-sign summary ===")
print(f"Valid rows:      {n_valid}")
print(f"Mismatches:      {n_mismatch} ({pct_mismatch:.1f}%)")
print("By direction:    ", counts)
print(f"\nSaved mismatches to: {out_csv}")

# If you want to look at it directly in Spyder:
mismatches_df.head()