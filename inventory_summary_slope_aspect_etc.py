"""
Spyder-friendly Landslide Inventory Summary (Clean Output, No Paragraph)

Usage:
1) Open this file in Spyder.
2) Press Run ▶. A file picker will appear (or hardcode CSV in choose_csv_path()).
3) Console prints a clean, labeled summary. Two files are written next to your CSV:
   - <csv>_summary_bullets.md   (bullet list summary)
   - <csv>_key_metrics.csv      (one-row metrics table)

Edit CONFIG below to override column names or thresholds.
"""

# =========================
# CONFIG (edit as needed)
# =========================
CONFIG = {
    "columns": {
        "slope": "ls_mean_slope",          # e.g., "ls_mean_slope"
        "aspect": "ls_mean_aspect",         # e.g., "ls_mean_aspect"
        "elevation": "ls_mean_height",      # e.g., "ls_mean_height" (m)
        "area": "ls_area_m2",           # e.g., "ls_area_m2" (m^2)
        "length": "ls_max_diameter_m",         # e.g., "ls_max_diameter_m"
        "width": "ls_min_diameter_m",          # e.g., "ls_min_diameter_m"
        "aspect_ratio": None,   # optional if length/width not present
        "velocity": "ts_linear_vel_myr",       # e.g., "ts_linear_vel_myr" (m/yr)
        "dist_coast_m": None,   # optional
        "dist_road_m": None,    # optional
    },
    "vel_thresh_cm_yr": 2.0,
    "area_unit": "m2",          # "m2" or "km2" (input unit of area column)
    "coast_thresh_m": 500.0,
    "road_thresh_m": None,
    "elongate_thresh": 2.0,
    "slump_hi": 2.5,
    "earthflow_hi": 4.5,

    # Print an octant distribution line (N, NE, ..., NW)
    "print_octants": True,
}

# =========================
# Implementation
# =========================
import os
import math
import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TK = True
except Exception:
    _HAS_TK = False

def choose_csv_path():
    # Hardcode here if you prefer (uncomment and set):
    return r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv"

def _guess_column(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        for c in cols:
            if key in c:
                return cols[c]
    return None

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def friendly_range(series: pd.Series, unit: str = "", as_int: bool = False) -> str:
    s = series.dropna().astype(float)
    if s.empty:
        return "N/A"
    lo, hi = np.nanmin(s), np.nanmax(s)
    if as_int:
        return f"{int(round(lo))}–{int(round(hi))}{unit}"
    else:
        return f"{lo:.2f}–{hi:.2f}{unit}"

def km2_range(series_m2: pd.Series) -> str:
    s = _coerce_numeric(series_m2).dropna()
    if s.empty:
        return "N/A"
    s_km2 = s / 1e6
    return friendly_range(s_km2, unit=" km²")

def aspect_to_compass_bucket(a_deg: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    a = (a_deg % 360 + 360) % 360
    idx = int((a + 22.5) // 45) % 8
    return dirs[idx]

def ew_ns_class(a_deg: float) -> str:
    a = (a_deg % 360 + 360) % 360
    axes = np.array([0, 90, 180, 270])
    nearest = axes[np.argmin(np.abs(((a - axes + 180) % 360) - 180))]
    return "EW" if nearest in (90, 270) else "NS"

def circular_mean_deg(angles_deg: pd.Series) -> float:
    angles = angles_deg.dropna().values.astype(float)
    if angles.size == 0:
        return float("nan")
    ang = np.deg2rad(angles)
    mean_angle = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
    return mean_angle

def classify_hwg(ar: float, slump_hi=2.5, earthflow_hi=4.5) -> str:
    if math.isnan(ar):
        return "unknown"
    if ar < slump_hi:
        return "slump"
    elif ar <= earthflow_hi:
        return "earthflow"
    else:
        return "complex"

def build_colmap(df: pd.DataFrame, cfg_cols: dict) -> dict:
    guesses = {
        "slope": ["slope", "slope_deg", "slope_degree", "ls_mean_slope"],
        "aspect": ["aspect", "azimuth", "orientation", "ls_mean_aspect", "ls_orientation_deg"],
        "elevation": ["elev", "elevation", "height", "ls_mean_height"],
        "area": ["area_km2", "area_m2", "ls_area_m2", "area"],
        "length": ["length", "long", "major", "max_diameter", "ls_max_diameter_m"],
        "width": ["width", "short", "minor", "min_diameter", "ls_min_diameter_m"],
        "aspect_ratio": ["ar", "axis_ratio", "aspect_ratio", "ls_axis_ratio"],
        "velocity": ["vel", "velocity", "ts_linear_vel_myr", "ts_dry1_vel_myr", "ts_dry2_vel_myr"],
        "dist_coast_m": ["dist_coast", "coast", "distance_to_coast", "dist_coast_m"],
        "dist_road_m": ["dist_road", "road", "distance_to_road", "dist_road_m"],
    }
    colmap = {}
    for k, cand in guesses.items():
        colmap[k] = cfg_cols.get(k) if cfg_cols.get(k) else _guess_column(df, cand)
    return colmap

def slope_bins_percent(slope_deg: pd.Series):
    """Return dict for bins:
       <19°, 19–25°, 26–35°, >35°
    """
    s = _coerce_numeric(slope_deg).dropna()
    if s.empty:
        return {}
    total = len(s)
    bins = {
        "<19°": (s < 19).sum(),
        "19–25°": ((s >= 19) & (s <= 25)).sum(),
        "26–35°": ((s >= 26) & (s <= 35)).sum(),
        ">35°": (s > 35).sum(),
    }
    return {k: (v, 100.0 * v / total) for k, v in bins.items()}

def main():
    # Choose CSV
    csv_path = choose_csv_path()
    if not csv_path or not os.path.exists(csv_path):
        print("No CSV selected or path not found. Edit choose_csv_path() to hardcode your file, or re-run and select a CSV.")
        return

    df = pd.read_csv(csv_path)

    colmap = build_colmap(df, CONFIG["columns"])
    print("Detected columns (override in CONFIG['columns'] if needed):")
    for k, v in colmap.items():
        print(f"  {k}: {v}")

    # Pull columns
    slope = _coerce_numeric(df[colmap["slope"]]) if colmap["slope"] else pd.Series(dtype=float)
    aspect = _coerce_numeric(df[colmap["aspect"]]) if colmap["aspect"] else pd.Series(dtype=float)
    elev = _coerce_numeric(df[colmap["elevation"]]) if colmap["elevation"] else pd.Series(dtype=float)
    area = _coerce_numeric(df[colmap["area"]]) if colmap["area"] else pd.Series(dtype=float)
    length = _coerce_numeric(df[colmap["length"]]) if colmap["length"] else pd.Series(dtype=float)
    width = _coerce_numeric(df[colmap["width"]]) if colmap["width"] else pd.Series(dtype=float)
    vel = _coerce_numeric(df[colmap["velocity"]]) if colmap["velocity"] else pd.Series(dtype=float)

    # Aspect ratio
    if not length.empty and not width.empty:
        with np.errstate(divide='ignore', invalid='ignore'):
            ar = (length / width).replace([np.inf, -np.inf], np.nan)
    elif colmap["aspect_ratio"]:
        ar = _coerce_numeric(df[colmap["aspect_ratio"]])
    else:
        ar = pd.Series(dtype=float)

    # Ranges
    n_total = len(df)
    slope_range = friendly_range(slope, unit="°")
    elev_range = friendly_range(elev, unit=" m", as_int=True)
    area_range = km2_range(area) if CONFIG["area_unit"].lower() == "m2" else friendly_range(area, unit=" km²")

    # Aspect stats
    circ_mean = circular_mean_deg(aspect) if not aspect.empty else float("nan")
    ew, ns = 0, 0
    octants_line = ""
    if not aspect.empty:
        classes = [ew_ns_class(a) for a in aspect.dropna()]
        total = len(classes)
        if total > 0:
            ew = round(100.0 * classes.count("EW") / total, 1)
            ns = round(100.0 * classes.count("NS") / total, 1)
        if CONFIG["print_octants"]:
            octants = [aspect_to_compass_bucket(a) for a in aspect.dropna()]
            vc = pd.Series(octants).value_counts(normalize=True).sort_index()
            octants_line = ", ".join([f"{k}:{round(v*100,1)}%" for k, v in vc.items()])

    # Activity above threshold
    active_line = "N/A"
    if not vel.empty:
        n_valid = int(vel.notna().sum())
        n_above = int((vel > (CONFIG["vel_thresh_cm_yr"]/100.0)).sum())
        if n_valid > 0:
            active_pct = 100.0 * n_above / n_valid
            active_line = f"{n_above}/{n_valid} ({active_pct:.1f}%) above {CONFIG['vel_thresh_cm_yr']:.1f} cm/yr"

    # Shape & Handwerger classes
    elong_round_line = "N/A"
    hwg_line = "N/A"
    if not ar.empty:
        arv = ar.dropna()
        if not arv.empty:
            elongate = int((arv >= CONFIG["elongate_thresh"]).sum())
            round_ = int((arv < CONFIG["elongate_thresh"]).sum())
            total_ar = int(arv.size)
            elong_round_line = f"{elongate}/{total_ar} elongate, {round_}/{total_ar} round (thr {CONFIG['elongate_thresh']:.1f})"
            hwg_classes = [classify_hwg(a, CONFIG["slump_hi"], CONFIG["earthflow_hi"]) for a in arv]
            vc = pd.Series(hwg_classes).value_counts(normalize=True)
            hwg_parts = [f"{k}:{int(round(vc.get(k,0)*100))}%" for k in ["slump","earthflow","complex"]]
            hwg_line = ", ".join(hwg_parts)

    # Slope bins
    slope_bins = slope_bins_percent(slope)
    bins_line = ""
    if slope_bins:
        bins_line = "; ".join([f"{k}: {v[0]} ({v[1]:.1f}%)" for k, v in slope_bins.items()])

    # Print clean summary
    print("\n=== Landslide Inventory Summary (Clean) ===\n")
    print(f"Count: {n_total}")
    if slope_range != "N/A": print(f"Slope range: {slope_range}")
    if elev_range != "N/A":  print(f"Elevation range: {elev_range}")
    if area_range != "N/A":  print(f"Area range: {area_range}")
    if not math.isnan(circ_mean): print(f"Aspect circular mean: {circ_mean:.1f}°")
    if not aspect.empty: print(f"Aspect EW vs NS: {ew}% EW vs {ns}% NS (excess {ew-ns:+.1f}%)")
    if octants_line: print(f"Aspect octants: {octants_line}")
    if elong_round_line != "N/A": print(f"Shape (AR): {elong_round_line}")
    if hwg_line != "N/A": print(f"Handwerger classes: {hwg_line}")
    if active_line != "N/A": print(f"Active above background: {active_line}")
    if bins_line: print(f"Slope bins: {bins_line}")

    # Save outputs
    base, _ = os.path.splitext(csv_path)
    md_path = base + "_summary_bullets.md"
    csv_out = base + "_key_metrics.csv"

    bullets = []
    bullets.append("# Landslide Inventory Summary (Clean)")
    bullets.append("")
    bullets.append(f"- **Count**: {n_total}")
    if slope_range != "N/A": bullets.append(f"- **Slope range**: {slope_range}")
    if elev_range != "N/A": bullets.append(f"- **Elevation range**: {elev_range}")
    if area_range != "N/A": bullets.append(f"- **Area range**: {area_range}")
    if not math.isnan(circ_mean): bullets.append(f"- **Aspect circular mean**: {circ_mean:.1f}°")
    if not aspect.empty: bullets.append(f"- **Aspect EW vs NS**: {ew}% EW vs {ns}% NS (excess {ew-ns:+.1f}%)")
    if octants_line: bullets.append(f"- **Aspect octants**: {octants_line}")
    if elong_round_line != "N/A": bullets.append(f"- **Shape (AR)**: {elong_round_line}")
    if hwg_line != "N/A": bullets.append(f"- **Handwerger classes**: {hwg_line}")
    if active_line != "N/A": bullets.append(f"- **Active above background**: {active_line}")
    if bins_line: bullets.append(f"- **Slope bins**: {bins_line}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(bullets))

    key_metrics = {
        "count_total": n_total,
        "slope_range_deg": slope_range,
        "elev_range_m": elev_range,
        "area_range_km2": area_range,
        "aspect_circ_mean_deg": None if math.isnan(circ_mean) else round(circ_mean,1),
        "ew_pct": ew,
        "ns_pct": ns,
        "ew_minus_ns_pct": None if aspect.empty else round(ew-ns,1),
        "elongate_vs_round": elong_round_line,
        "handwerger_classes": hwg_line,
        "active_above_thresh": active_line,
        "slope_bins": bins_line,
        "aspect_octants": octants_line,
    }
    pd.DataFrame([key_metrics]).to_csv(csv_out, index=False)

    print(f"\nSaved:\n  - {md_path}\n  - {csv_out}\n")

if __name__ == "__main__":
    main()