#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalized LOS displacement + cumulative precipitation + cumulative PGA
by user-defined groups (e.g., velocity-ratio, PGA-ratio, precipitation-ratio).
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import h5py

from NC_Landslides_paths import *

# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 2
vel_multiple      = 5
NORM_METHOD = "minmax"
SMOOTH_WIN  = 5
BAND_ALPHA = 0.2

# Earthquake dates
eq1 = 2021.9685
eq2 = 2022.9685

CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"

# -------------------------
# Helpers
# -------------------------
def load_timeseries(fpath):
    with h5py.File(fpath, "r") as hf:
        dates = hf["dates"][:]
        ts    = hf["clean_ts"][:]
        sign  = hf["meta"].attrs.get("ls_sign", 1.0)
    return dates, ts, sign

def load_drivers(fpath):
    """Load cumulative 14-day precipitation and PGA"""
    with h5py.File(fpath, "r") as hf:
        if "dates_14day_decimal" not in hf:
            return None, None, None
        dates = hf["dates_14day_decimal"][:]

        precip = hf["/rainfall/rain_14day_cum"][:] if "/rainfall/rain_14day_cum" in hf else None
        pga    = hf["pga/pga_14day_cum"][:] if "pga/pga_14day_cum" in hf else None
        return dates, precip, pga
    return None, None, None

def normalize_series(vals, sign=1.0, method="minmax"):
    ts0 = (vals - np.nanmean(vals)) * sign
    if method == "minmax":
        mn, mx = np.nanmin(ts0), np.nanmax(ts0)
        normed = (ts0 - mn) / (mx - mn) if mx > mn else np.zeros_like(ts0)
    elif method == "zscore":
        s = np.nanstd(ts0)
        normed = (ts0 - np.nanmean(ts0)) / s if s > 0 else np.zeros_like(ts0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normed

def summarize_panel(df_paths, method="minmax", window=5):
    disp_series, rain_series, pga_series = [], [], []
    n_failed = 0

    for _, row in df_paths.iterrows():
        fpath = row["file"]
        try:
            # Displacement
            dates, ts, sign = load_timeseries(fpath)
            disp_series.append(pd.Series(normalize_series(ts, sign, method),
                                         index=dates, name=row["ls_id"]))

            # Precip & PGA
            ddates, precip, pga = load_drivers(fpath)
            if ddates is not None and precip is not None:
                norm_rain = normalize_series(precip, 1.0, method)
                rain_series.append(pd.Series(norm_rain, index=ddates, name=row["ls_id"]))
            if ddates is not None and pga is not None:
                norm_pga = normalize_series(pga, 1.0, method)
                pga_series.append(pd.Series(norm_pga, index=ddates, name=row["ls_id"]))
        except Exception as e:
            print(f"[WARN] Failed loading {fpath}: {e}")
            n_failed += 1
            continue

    if len(disp_series) == 0:
        return None, None, None, None, 0, 0, n_failed

    # Combine displacement
    df_disp = pd.concat(disp_series, axis=1)
    long_disp = df_disp.stack().reset_index()
    long_disp.columns = ["date", "ls_id", "value"]

    summary_disp = long_disp.groupby("date")["value"].agg(
        median=lambda x: np.nanmedian(x),
        p5=lambda x: np.nanpercentile(x, 5),
        p95=lambda x: np.nanpercentile(x, 95),
    ).sort_index()
    summary_disp["med_sm"] = summary_disp["median"].rolling(window, center=True, min_periods=1).mean()
    summary_disp["p5_sm"]  = summary_disp["p5"].rolling(window, center=True, min_periods=1).mean()
    summary_disp["p95_sm"] = summary_disp["p95"].rolling(window, center=True, min_periods=1).mean()

    # Aggregate rainfall
    summary_rain = None
    if len(rain_series) > 0:
        df_rain = pd.concat(rain_series, axis=1)
        summary_rain = pd.DataFrame({"median": df_rain.median(axis=1)}).sort_index()

    # Aggregate PGA
    summary_pga = None
    if len(pga_series) > 0:
        df_pga = pd.concat(pga_series, axis=1)
        summary_pga = pd.DataFrame({"median": df_pga.median(axis=1)}).sort_index()

    # Metrics for displacement
    n_series = df_disp.shape[1]
    n_points = (~df_disp.isna()).sum().sum()

    return long_disp, summary_disp, summary_rain, summary_pga, n_series, int(n_points), n_failed

def plot_panel(ax, long_disp, summary_disp, summary_rain, summary_pga,
               title, n_series, n_failed):

    if long_disp is None or summary_disp is None:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return

    ax.grid(False)

    # EQ markers
    ax.axvline(eq1, color="red", linestyle="--", alpha=0.5, lw=1.5, zorder=1)
    ax.axvline(eq2, color="red", linestyle="--", alpha=0.5, lw=1.5, zorder=1)

    # 5–95% displacement band
    ax.fill_between(summary_disp.index, summary_disp["p5_sm"], summary_disp["p95_sm"],
                    color="steelblue", alpha=BAND_ALPHA, zorder=3)

    # Precipitation
    if summary_rain is not None:
        ax.plot(summary_rain.index, summary_rain["median"], lw=1.8, color="royalblue", zorder=4)

    # PGA
    if summary_pga is not None:
        ax.plot(summary_pga.index, summary_pga["median"], lw=1.8, color="purple", zorder=5)
        
    # Median displacement
    ax.plot(summary_disp.index, summary_disp["median"],
            lw=1.2, color="black", linestyle="--", zorder=6)

    # Smoothed displacement
    ax.plot(summary_disp.index, summary_disp["med_sm"],
            lw=2.0, color="darkorange", zorder=7)

    # Unified y-axis
    ax.set_ylabel("Normalized Data")
    ax.set_ylim(-0.05, 1.05)

    # Badge
    badge = f"N={n_series}"
    ax.text(0.02, 0.98, badge, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=1))

    ax.set_title(title, pad=6, fontsize=11)
    ax.set_xlabel("Decimal Year")

# -------------------------
# Grouping strategies
# -------------------------
def group_by_velocity_ratio(df, vel_multiple=5):
    df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]
    def categorize(ratio):
        if ratio < 1 / vel_multiple: return "Much Slower"
        elif 1 / vel_multiple <= ratio < 0.83: return "Slower"
        elif 0.83 <= ratio <= 1.2: return "Similar"
        elif 1.2 < ratio <= vel_multiple: return "Faster"
        else: return "Much Faster"
    df["group"] = df["vel_ratio"].apply(categorize)
    return df

# -------------------------
# Data load + grouping
# -------------------------
df = pd.read_csv(CSV_PATH)
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
df = df[(df["vel_dry1"] >= vel_min_threshold) | (df["vel_dry2"] >= vel_min_threshold)].copy()

# choose grouping strategy here:
df = group_by_velocity_ratio(df, vel_multiple=vel_multiple)

# Map HDF5 files
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
    raise RuntimeError("Could not find a landslide ID column in the CSV")
csv_id_col = id_candidates[0]
df[csv_id_col] = df[csv_id_col].astype(str)

merged = df.merge(df_h5, left_on=csv_id_col, right_on="ls_id", how="left")

# -------------------------
# Panels
# -------------------------
panel_groups = list(df["group"].dropna().unique()) + ["Active (exclude Much Slower)"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=False, sharey=False)
axes = axes.flatten()

results = []
for ax, name in zip(axes, panel_groups):
    if name == "Active (exclude Much Slower)":
        sub = merged[(merged["group"] != "Much Slower") & merged["file"].notna()][["ls_id","file"]].copy()
    else:
        sub = merged[(merged["group"] == name) & merged["file"].notna()][["ls_id","file"]].copy()

    long_disp, summary_disp, summary_rain, summary_pga, n_series, n_points, n_failed = summarize_panel(
        sub, method=NORM_METHOD, window=SMOOTH_WIN)
    plot_panel(ax, long_disp, summary_disp, summary_rain, summary_pga, name, n_series, n_failed)
    results.append({"group": name, "N": n_series, "n_failed": n_failed})

for ax, lab in zip(axes, ['a','b','c','d','e','f']):
    ax.text(0.02, 1.065, f'{lab})', transform=ax.transAxes, ha='left', va='top', fontsize=12)

legend_handles = [
    Line2D([0],[0], lw=2.0, color="darkorange", label='Smoothed disp.'),
    Line2D([0],[0], lw=1.2, color="black", linestyle="--", label='Median disp'),
    Patch(facecolor="steelblue", alpha=BAND_ALPHA, label='5–95% disp'),
    Line2D([0],[0], lw=1.8, color="royalblue", label='Median cum. precip.'),
    Line2D([0],[0], lw=1.8, color="purple", label='Median cum. PGA'),
    Line2D([0],[0], color="red", alpha=0.5, linestyle="--", lw=1.5, label='Earthquakes'),
]
axes[-1].legend(handles=legend_handles, loc="lower right", frameon=True)

fig.suptitle(
    f"Normalized LOS displacement + 14-day cumulative precipitation + PGA by groups (min≥{vel_min_threshold} cm/yr; ×{vel_multiple}; norm={NORM_METHOD})",
    fontsize=14, y=0.995
)
plt.tight_layout(rect=[0, 0, 1, 0.98])

out_png = os.path.join(fig_dir, f"norm_ts_with_14dayprecipPGA_x{vel_multiple}_min{vel_min_threshold}_{NORM_METHOD}.png")
out_pdf = out_png.replace(".png", ".pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

results_df = pd.DataFrame(results)
out_csv = os.path.join(fig_dir, f"norm_ts_metrics_with_14dayprecipPGA_x{vel_multiple}_min{vel_min_threshold}_{NORM_METHOD}.csv")
results_df.to_csv(out_csv, index=False)

print(f"→ saved: {out_png}, {out_pdf}, {out_csv}")
