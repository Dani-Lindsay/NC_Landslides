#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 20:55:19 2025

@author: daniellelindsay
"""

# Minimal per-source + overall detection summary
# - Set CSV_PATH to your file
# - Requires pandas

import pandas as pd
import numpy as np
import re

# ---- EDIT THIS ----
CSV_PATH = r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv"

SOURCE_COL = "sources"
SELECTED_COL = "selected"
# -------------------

# Read data
df = pd.read_csv(CSV_PATH)

# --- Make 'selected' a boolean (True/False/NaN) ---
TRUE_SET  = {"true","t","1","yes","y","on"}
FALSE_SET = {"false","f","0","no","n","off"}

def to_bool(x):
    if isinstance(x, (bool, np.bool_)):  # already boolean
        return bool(x)
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in TRUE_SET:  return True
    if s in FALSE_SET: return False
    # try numeric 1/0
    try:
        v = float(s)
        if v == 1: return True
        if v == 0: return False
    except Exception:
        pass
    return np.nan

sel_bool = df[SELECTED_COL].apply(to_bool)

# --- Overall kept (no double counting) ---
total_candidates   = len(df)
with_selected_flag = int(sel_bool.notna().sum())
kept_count         = int((sel_bool == True).sum())
not_detected_count = int((sel_bool == False).sum())
kept_percent       = (100.0 * kept_count / with_selected_flag) if with_selected_flag > 0 else np.nan

print("\n=== Overall kept (selected==True) ===")
print(f"total_candidates      = {total_candidates}")
print(f"with_selected_flag    = {with_selected_flag}")
print(f"kept_count            = {kept_count}")
print(f"not_detected_count    = {not_detected_count}")
print(f"kept_percent          = {'N/A' if np.isnan(kept_percent) else f'{kept_percent:.1f}%'}")

# --- Per-source detection (handles multi-source rows) ---
# Split sources on common delimiters: ; , | / & and whitespace around them
split_re = re.compile(r"[;,&/|]+")

def split_sources(s):
    if pd.isna(s):
        return []
    parts = []
    for chunk in split_re.split(str(s)):
        for sub in chunk.split(","):
            name = sub.strip()
            if name:
                parts.append(name)
    return parts

exploded = df.copy()
exploded["__sources_list"] = df[SOURCE_COL].apply(split_sources)
exploded["__selected_bool"] = sel_bool
exploded = exploded.explode("__sources_list", ignore_index=True)
exploded = exploded.rename(columns={"__sources_list": "source"})
exploded = exploded.dropna(subset=["source"])

# Group by source and compute stats
rows = []
for src, g in exploded.groupby("source", dropna=False):
    total = len(g)
    valid = g["__selected_bool"].dropna()
    n_valid = int(valid.size)
    n_active = int((valid == True).sum())
    n_inactive = int((valid == False).sum())
    pct_active = (100.0 * n_active / n_valid) if n_valid > 0 else np.nan
    rows.append({
        "source": src,
        "total_candidates": total,
        "with_selected_flag": n_valid,
        "active_count": n_active,
        "not_detected_count": n_inactive,
        "active_percent": np.nan if np.isnan(pct_active) else round(pct_active, 1),
    })

out = pd.DataFrame(rows).sort_values(
    ["active_percent", "total_candidates"], ascending=[False, False]
).reset_index(drop=True)

print("\n=== Detection by Source ===")
for _, r in out.iterrows():
    pct = r['active_percent']
    pct_str = "N/A" if pd.isna(pct) else f"{pct:.1f}%"
    print(f"{r['source']}: total={int(r['total_candidates'])}, "
          f"with_selected={int(r['with_selected_flag'])}, "
          f"active={int(r['active_count'])}, "
          f"not_detected={int(r['not_detected_count'])}, "
          f"active%={pct_str}")

# If you want a CSV output too, uncomment:
# out.to_csv(CSV_PATH.replace(".csv", "_source_detection_summary.csv"), index=False)
