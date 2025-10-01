#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoothed median comparison across grouping strategies
(velocity ratio, PGA ratio, precipitation ratio, area),
A4-ready (portrait), font size 10, plasma colormap.
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from NC_Landslides_paths import *

# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 2
vel_multiple      = 5
NORM_METHOD = "minmax"
SMOOTH_WIN  = 5

# Earthquake dates (decimal years)
eq1 = 2021.9685   # 20 Dec 2021
eq2 = 2022.9685   # 20 Dec 2022

CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"

# Colormap: sequential
CMAP = plt.get_cmap("plasma")

# -------------------------
# Helpers
# -------------------------
def load_timeseries(fpath):
    with h5py.File(fpath, "r") as hf:
        dates = hf["dates"][:]
        ts    = hf["clean_ts"][:]
        sign  = hf["meta"].attrs.get("ls_sign", 1.0)
    return dates, ts, sign

def normalize_series(vals, sign=1.0, method="minmax"):
    ts0 = (vals - np.nanmean(vals)) * sign
    if method == "minmax":
        mn, mx = np.nanmin(ts0), np.nanmax(ts0)
        return (ts0 - mn) / (mx - mn) if mx > mn else np.zeros_like(ts0)
    elif method == "zscore":
        s = np.nanstd(ts0)
        return (ts0 - np.nanmean(ts0)) / s if s > 0 else np.zeros_like(ts0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def summarize_panel(df_paths, method="minmax", window=5):
    """Return smoothed median displacement for a subset of landslides."""
    disp_series = []
    for _, row in df_paths.iterrows():
        try:
            dates, ts, sign = load_timeseries(row["file"])
            disp_series.append(
                pd.Series(normalize_series(ts, sign, method), index=dates, name=row["ls_id"])
            )
        except Exception as e:
            print(f"[WARN] Failed loading {row['file']}: {e}")
            continue

    if len(disp_series) == 0:
        return None

    df_disp = pd.concat(disp_series, axis=1)
    long_disp = df_disp.stack().reset_index()
    long_disp.columns = ["date", "ls_id", "value"]

    # Robust to pandas versions: compute Series, then to_frame(name="median")
    median_series = long_disp.groupby("date")["value"].median()
    summary_disp = median_series.to_frame(name="median").sort_index()

    summary_disp["med_sm"] = summary_disp["median"].rolling(window, center=True, min_periods=1).mean()
    return summary_disp

# -------------------------
# Grouping functions
# -------------------------
def group_by_velocity_ratio(df, vel_multiple=5):
    df = df.copy()
    df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]
    def categorize(ratio):
        if ratio < 1 / vel_multiple: return "Much Slower"
        elif 1 / vel_multiple <= ratio < 0.83: return "Slower"
        elif 0.83 <= ratio <= 1.2: return "Similar"
        elif 1.2 < ratio <= vel_multiple: return "Faster"
        else: return "Much Faster"
    df["group"] = df["vel_ratio"].apply(categorize)
    return df

def group_by_pga_ratio(df):
    df = df.copy()
    df["pga_ratio"] = df["support_params/wy23_vs_wy22_pga_ratio"]
    def categorize(ratio):
        if ratio < 0.6: return "Much Lower"
        elif 0.6 <= ratio < 0.83: return "Lower"
        elif 0.83 <= ratio <= 1.2: return "Similar"
        elif 1.2 < ratio <= 3: return "Higher"
        else: return "Much Higher"
    df["group"] = df["pga_ratio"].apply(categorize)
    return df

def group_by_precip_ratio(df):
    df = df.copy()
    df["precip_ratio"] = df["support_params/wy23_vs_wy22_rain_ratio"]
    def categorize(ratio):
        if ratio <= 1.1: return "Similar"
        elif 1.1 < ratio <= 1.5: return "Higher"
        else: return "Much Higher"
    df["group"] = df["precip_ratio"].apply(categorize)
    return df

def group_by_area(df):
    """
    Categorize landslide area into 5 bins (merge the two smallest):
      • Small (<=3e4 m²)
      • Medium-Small (3e4–1e5 m²)
      • Medium (1e5–3e5 m²)
      • Large (3e5–1e6 m²)
      • Largest (>=1e6 m²)
    """
    df = df.copy()

    def categorize(area_m2):
        if pd.isna(area_m2) or area_m2 <= 0:
            return "Unknown"
        log_area = np.log10(area_m2)

        # merged smallest two bins
        if log_area < 4.5:                 # < 10^4.5  (<= 3e4 m²)
            return "Smallest (≤3e4 m²)"
        elif 4.5 <= log_area < 5.0:         # 3e4–1e5
            return "Small (3e4–1e5 m²)"
        elif 5.0 <= log_area < 5.5:         # 1e5–3e5
            return "Medium (1e5–3e5 m²)"
        elif 5.5 <= log_area < 6.0:         # 3e5–1e6
            return "Large (3e5–1e6 m²)"
        else:                               # ≥ 1e6
            return "Largest (≥1e6 m²)"

    df["group"] = df["meta__ls_area_m2"].apply(categorize)
    return df

GROUPING_STRATEGIES = {
    "Velocity Ratio": group_by_velocity_ratio,
    "PGA Ratio":      group_by_pga_ratio,
    "Precipatation Ratio":   group_by_precip_ratio,
    "Landslide Area":     group_by_area,
}

# -------------------------
# Plotting
# -------------------------
def plot_group_comparisons(df, groupings, eq_dates, group_orders=None):
    # A4-friendly text
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))  # A4 portrait
    axes = axes.flatten()

    # Prepare HDF5 map once (faster)
    h5_files = glob.glob(os.path.join(H5_DIR, "*.h5"))
    h5_records = []
    for fp in h5_files:
        try:
            with h5py.File(fp, "r") as hf:
                sid = hf["meta"].attrs.get("ID")
                h5_records.append({"ls_id": str(sid), "file": fp})
        except Exception:
            continue
    df_h5 = pd.DataFrame(h5_records).dropna()

    id_candidates = [c for c in df.columns if re.search(r"(ls_id|meta__ls_id|ID)$", c)]
    if not id_candidates:
        raise RuntimeError("Could not find a landslide ID column")
    csv_id_col = id_candidates[0]

    for ax, (name, func) in zip(axes, groupings.items()):
        grouped = func(df.copy())
        grouped[csv_id_col] = grouped[csv_id_col].astype(str)
        merged = grouped.merge(df_h5, left_on=csv_id_col, right_on="ls_id", how="left")

        # custom group order if provided
        if group_orders and name in group_orders:
            group_list = [g for g in group_orders[name] if g in merged["group"].unique()]
        else:
            group_list = sorted(merged["group"].dropna().unique())

        # sequential colors in the specified order
        colors = CMAP(np.linspace(0, 1, len(group_list)))

        for color, g in zip(colors, group_list):
            sub = merged[(merged["group"] == g) & merged["file"].notna()][["ls_id", "file"]].copy()
            summary = summarize_panel(sub, method=NORM_METHOD, window=SMOOTH_WIN)
            if summary is None:
                continue
            # normalize each median curve to 0–1
            vals = summary["med_sm"].values
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax > vmin:
                vals = (vals - vmin) / (vmax - vmin)
            ax.plot(summary.index, vals, label=g, color=color, lw=2)

        # earthquake markers
        for eq in eq_dates:
            ax.axvline(eq, color="black", linestyle="--", alpha=0.5, lw=1.2)

        ax.set_title(name.capitalize())
        ax.set_xlabel("Decimal Year")
        ax.set_ylabel("Normalized displacement")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle("Smoothed median comparison across grouping strategies", fontsize=12, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = os.path.join(fig_dir, "norm_ts_all_groupings_comparison.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"→ saved: {out_png}, {out_pdf}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
    df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
    df = df[(df["vel_dry1"] >= vel_min_threshold) | (df["vel_dry2"] >= vel_min_threshold)].copy()
    df["meta__ls_area_m2_log10"] = np.log10(df["meta__ls_area_m2"])

    # custom group orders
    VELOCITY_ORDER = ["Much Faster", "Faster", "Similar", "Slower", "Much Slower"]
    PGA_ORDER      = ["Much Higher", "Higher", "Similar", "Lower", "Much Lower"]
    PRECIP_ORDER   = ["Similar", "Higher", "Much Higher"]
    AREA_ORDER = [
    "Smallest (≤3e4 m²)",
    "Small (3e4–1e5 m²)",
    "Medium (1e5–3e5 m²)",
    "Large (3e5–1e6 m²)",
    "Largest (≥1e6 m²)",
]
    
    GROUP_ORDERS = {
        "Velocity Ratio": VELOCITY_ORDER,
        "PGA Ratio":      PGA_ORDER,
        "Precipatation Ratio":   PRECIP_ORDER,
        "Landslide Area":     AREA_ORDER,
    }

    plot_group_comparisons(
        df,
        groupings=GROUPING_STRATEGIES,
        eq_dates=[eq1, eq2],
        group_orders=GROUP_ORDERS,
    )
