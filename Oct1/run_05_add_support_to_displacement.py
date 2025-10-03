#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:56:38 2025

Augments displacement HDF5 with supporting attributes and resamples supporting
data to match a 14-day interval starting from the first InSAR acquisition date.

- Retains existing meta and params groups
- Adds support_meta and support_params
- Adds dates_14day and dates_14day_decimal
- Adds new groups: geology, pga, rainfall

Not felt: PGA < 0.0005 g

Weak: 0.003 ≤ PGA < 0.0276 g

Light: 0.0276 ≤ PGA < 0.062 g

@author: daniellelindsay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd

from hdf5_support import load_landslide_hdf5, _save_item
from landslide_utlities import date_to_decimal_year

import warnings
from NC_Landslides_paths import *
warnings.filterwarnings("ignore", message=".*errors='ignore'.*")


# ---------------- User paths ----------------
displacement_dir = "/Volumes/Seagate/NC_Landslides/Data_3/LS_Timeseries"
supporting_dir   = "/Volumes/Seagate/NC_Landslides/Inputs/landslide_supporting"

# ---------------- Settings ----------------
PGA_THRESHOLD_G = 0.001          # “felt” threshold
WY_YEARS_TO_SAVE = [2022, 2023]  # water years to summarize
OUTPUT_CSV = os.path.join(ts_final_dir, "final_selection_only_mapped_pga_precip.csv")

# Map displacement ls_id -> supporting nearest_event_id from CSV (loaded once)
MAPPING_CSV = os.path.join(ts_final_dir, "final_selection_only_mapped.csv")

try:
    _map_df = pd.read_csv(MAPPING_CSV, dtype={"ls_id": str, "support_id": str})
    _map_df = _map_df[["ls_id", "support_id"]].dropna()
    _ID_TO_SUPPORT = dict(zip(_map_df["ls_id"].str.strip(),
                              _map_df["support_id"].str.strip()))
except Exception as e:
    print(f"! Could not load mapping CSV ({MAPPING_CSV}): {e}")
    _ID_TO_SUPPORT = {}

# ---------------- Helpers ----------------
def decimal_year_to_datetime(decimal_years):
    """Convert decimal year(s) -> pandas datetime index."""
    if isinstance(decimal_years, (float, int, np.floating, np.integer)):
        decimal_years = [float(decimal_years)]
    out = []
    for y in decimal_years:
        y = float(y)
        year = int(y)
        frac = y - year
        start = datetime(year, 1, 1)
        end   = datetime(year + 1, 1, 1)
        out.append(start + timedelta(seconds=frac * (end - start).total_seconds()))
    return pd.to_datetime(out)

def get_id_from_disp_filename(path):
    """Extract landslide id like 'ls_001' from displacement filename."""
    m = re.match(r"(ls_\d+)", os.path.basename(path))
    return m.group(1) if m else None

# def supporting_path_for(disp_path):
#     slid = get_id_from_disp_filename(disp_path)
#     return os.path.join(supporting_dir, f"{slid}-supporting.h5") if slid else None

def supporting_path_for(disp_path):
    """Return supporting H5 path using CSV-mapped nearest_event_id for this displacement file."""
    slid = get_id_from_disp_filename(disp_path)  # e.g., "ls_001"
    supp_id = _ID_TO_SUPPORT.get(slid) if slid else None
    if not supp_id:  # no mapping → let caller skip
        return None
    return os.path.join(supporting_dir, f"{supp_id}-supporting.h5")

def ensure_dataset(group, name, array_like):
    if name in group:
        del group[name]
    group[name] = np.array(array_like)

def water_year_mask(dates, wy_year):
    """Oct 1 (wy-1) .. Sep 30 (wy)."""
    start = pd.Timestamp(f"{wy_year-1}-10-01")
    end   = pd.Timestamp(f"{wy_year}-09-30 23:59:59.999999")
    return (dates >= start) & (dates <= end)

def safe_ratio(new, old):
    """Return ratio new/old with safe handling of zeros."""
    if np.isnan(new) or np.isnan(old):
        return np.nan
    if old == 0:
        return np.inf if new > 0 else 1.0
    return float(new) / float(old)

def pct_change(new, old):
    """Percent change (new-old)/|old|*100, robust to old==0."""
    if np.isnan(new) or np.isnan(old):
        return np.nan
    if old == 0:
        return 100.0 if new > 0 else 0.0
    return float(new - old) / abs(float(old)) * 100.0

# ---------------- Core processing ----------------
def process_pair(disp_path, supp_path):
    """Process one (displacement, supporting) pair in-place. Returns row dict for CSV."""
    row = {"file": os.path.basename(disp_path), "id": get_id_from_disp_filename(disp_path)}
    support_data = load_landslide_hdf5(supp_path)

    with h5py.File(disp_path, "r+") as disp:
        # --- Displacement dates (decimal year) -> datetime
        if "dates" not in disp:
            print(f"  ! Skipping (no /dates): {disp_path}")
            return None
        dec_years = np.asarray(disp["dates"][:], dtype=float)
        disp_dates = decimal_year_to_datetime(dec_years)
        insar_dates = pd.date_range(start=disp_dates[0], end=disp_dates[-1], freq="14D")

        # Save 14-day dates
        ensure_dataset(disp, "dates_14day", [d.strftime("%Y%m%d").encode("utf-8") for d in insar_dates])
        ensure_dataset(disp, "dates_14day_decimal", [date_to_decimal_year(d) for d in insar_dates])

        # --- Support meta/params (don’t touch original /meta or /params)
        if "support_meta" in disp:
            del disp["support_meta"]
        _save_item(disp, "support_meta", {
            "eq_event_radius": support_data["meta"]["eq_event_radius"],
            "support_lat": support_data["meta"]["lat"],
            "support_lon": support_data["meta"]["lon"],
            })

        if "support_params" in disp:
            del disp["support_params"]
        _save_item(disp, "support_params", {
            "dist_ocean_m": support_data["distances"]["ocean_m"],
            "dist_road_m":  support_data["distances"]["road_m"]
        })
        
        print(float(support_data["distances"].get("ocean_m", np.nan)))
        #print(float(support_data["distances"].get("road_m", np.nan)))
        
        row["dist_ocean_m"] = float(support_data["distances"].get("ocean_m", np.nan))
        #row["dist_road_m"]  = float(support_data["distances"].get("road_m",  np.nan))

        # --- Copy full groups from supporting
        for grp in ["geology", "pga", "rainfall"]:
            if grp in disp:
                del disp[grp]
            _save_item(disp, grp, support_data[grp])

        # --- Resample rainfall (14d)
        rain_dates = pd.to_datetime(support_data["rainfall"]["date"]).tz_localize(None)
        rain_mm    = np.asarray(support_data["rainfall"]["rain_mm"], dtype=float)

        rain_14day = []
        for t in insar_dates:
            window_end = t
            window_start = t - timedelta(days=14)
            m = (rain_dates > window_start) & (rain_dates <= window_end)
            rain_14day.append(np.nansum(rain_mm[m]) if np.any(m) else 0.0)
        rain_14day_cum = np.nancumsum(rain_14day)

        ensure_dataset(disp["rainfall"], "rain_14day",     rain_14day)
        ensure_dataset(disp["rainfall"], "rain_14day_cum", rain_14day_cum)

        # --- Resample PGA (14d)
        pga_time = pd.to_datetime(support_data["pga"]["event_time"]).tz_localize(None)
        pga_mean = np.asarray(support_data["pga"]["pga_mean"], dtype=float)

        pga_14day, eq_count = [], []
        for t in insar_dates:
            window_end = t
            window_start = t - timedelta(days=14)
            m = (pga_time > window_start) & (pga_time <= window_end)
            if np.any(m):
                pga_14day.append(np.nansum(pga_mean[m]))
                eq_count.append(int(np.sum(pga_mean[m] > PGA_THRESHOLD_G)))
            else:
                pga_14day.append(0.0)
                eq_count.append(0)
        pga_14day_cum = np.nancumsum(pga_14day)
        eq_count_cum  = np.cumsum(eq_count)

        ensure_dataset(disp["pga"], "pga_14day",      pga_14day)
        ensure_dataset(disp["pga"], "pga_14day_cum",  pga_14day_cum)
        ensure_dataset(disp["pga"], "eq_count",       eq_count)
        ensure_dataset(disp["pga"], "eq_count_cum",   eq_count_cum)

        # --- Water-year summaries on 14d series
        rain_arr = np.asarray(rain_14day, dtype=float)
        pga_arr  = np.asarray(pga_14day, dtype=float)
        eq_arr   = np.asarray(eq_count, dtype=int)

        wy = {}
        for y in WY_YEARS_TO_SAVE:
            m = water_year_mask(insar_dates, y)
            if np.any(m):
                wy[f"wy{y%100:02d}_rain_mm"]  = float(np.nansum(rain_arr[m]))
                wy[f"wy{y%100:02d}_pga"]      = float(np.nansum(pga_arr[m]))
                wy[f"wy{y%100:02d}_eq_count"] = int(np.nansum(eq_arr[m]))
            else:
                wy[f"wy{y%100:02d}_rain_mm"]  = np.nan
                wy[f"wy{y%100:02d}_pga"]      = np.nan
                wy[f"wy{y%100:02d}_eq_count"] = np.nan

        # --- WY23 vs WY22 diagnostics
        r22, r23 = wy.get("wy22_rain_mm", np.nan),  wy.get("wy23_rain_mm", np.nan)
        g22, g23 = wy.get("wy22_pga", np.nan),      wy.get("wy23_pga", np.nan)
        n22, n23 = wy.get("wy22_eq_count", np.nan), wy.get("wy23_eq_count", np.nan)

        diag = {
            "wy23_vs_wy22_rain_ratio": safe_ratio(r23, r22),
            "wy23_vs_wy22_rain_pct":   pct_change(r23, r22),
            "wy23_vs_wy22_pga_ratio":  safe_ratio(g23, g22),
            "wy23_vs_wy22_pga_pct":    pct_change(g23, g22),
            "wy23_vs_wy22_eq_ratio":   safe_ratio(n23, n22),
            "wy23_vs_wy22_eq_pct":     pct_change(n23, n22),
        }
        # ≥30% increase flags (robust to baseline==0)
        diag["wy23_more_rain_30pct"]    = int((r23 > r22 * 1.3) or (r22 == 0 and r23 > 0))
        diag["wy23_more_shaking_30pct"] = int((g23 > g22 * 1.3) or (g22 == 0 and g23 > 0))
        diag["wy23_more_eqs_30pct"]     = int((n23 > n22 * 1.3) or (n22 == 0 and n23 > 0))

        # Write WY summaries + diagnostics into /support_params
        spg = disp["support_params"]
        for k, v in {**wy, **diag}.items():
            ensure_dataset(spg, k, v)

        # Row for CSV summary
        row.update(wy)
        row.update(diag)

        # Quick console glance
        print(f"✓ {row['id']}") #": WY22 rain={r22:.1f}, WY23 rain={r23:.1f}, "
              #f"Δrain%={row['wy23_vs_wy22_rain_pct']:.1f}; "
              #f"WY22 pga={g22:.4f}, WY23 pga={g23:.4f}, "
              #f"Δpga%={row['wy23_vs_wy22_pga_pct']:.1f}; "
              #f"WY22 eqs={n22}, WY23 eqs={n23}, "
              #f"Δeq%={row['wy23_vs_wy22_eq_pct']:.1f}")
    return row

# ---------------- Main ----------------
def main():
    # discover displacement files
    disp_files = []
    for root, _, files in os.walk(displacement_dir):
        for f in files:
            if f.endswith(".h5") and "Timeseries" in f:
                disp_files.append(os.path.join(root, f))
    disp_files.sort()

    if not disp_files:
        print("No displacement files found.")
        return

    rows = []
    for dpath in disp_files:
        spath = supporting_path_for(dpath)
        if not spath or not os.path.exists(spath):
            print(f"! Skipping (no matching supporting): {os.path.basename(dpath)}")
            continue
        try:
            row = process_pair(dpath, spath)
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"! Error processing {os.path.basename(dpath)}: {e}")

    if not rows:
        print("No rows to summarize.")
        return

    # Build CSV summary
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote summary: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
