#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-figure (2 columns × 3 rows) normalized time-series by PGA-ratio group,
with shaded dry-season velocity windows and subplot labels.
"""

# =========================
# Imports & config
# =========================
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import h5py

from NC_Landslides_paths import *  # provides fig_dir etc.

# -------------------------
# Parameters
# -------------------------
vel_min_threshold = 2     # cm/yr threshold for inclusion (>= in either year)
NORM_METHOD = "minmax"    # "minmax" or "zscore"
SMOOTH_WIN  = 5           # rolling window (samples) for per-date summary

# Dry-season velocity windows (decimal year)
vel_t1 = 2022.1667
vel_t2 = 2022.9167
vel_t3 = 2023.1667
vel_t4 = 2023.9167

SPAN_WY22_COLOR = "#e6e6e6"
SPAN_WY23_COLOR = "#e6e6e6"
BAND_ALPHA = 0.25

# Data locations
CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data_1/LS_Final_TS_4/compiled_landslide_data.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data_1/LS_Final_TS_4"

CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data_3/LS_Timeseries/final_selection_only_with_pga_precip.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data_3/LS_Timeseries"

# =========================
# Helpers
# =========================
def load_timeseries(fpath):
    with h5py.File(fpath, "r") as hf:
        dates = hf["dates"][:]           # decimal-year
        ts    = hf["clean_ts"][:]        # LOS in meters
        sign  = hf["meta"].attrs.get("ls_sign", 1.0)
    return dates, ts, sign

def normalize_ts(dates, ts, sign=1.0, method="minmax"):
    ts0 = (ts - np.nanmean(ts)) * sign
    if method == "minmax":
        mn, mx = np.nanmin(ts0), np.nanmax(ts0)
        normed = (ts0 - mn) / (mx - mn) if mx > mn else np.zeros_like(ts0)
    elif method == "zscore":
        s = np.nanstd(ts0)
        normed = (ts0 - np.nanmean(ts0)) / s if s > 0 else np.zeros_like(ts0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return pd.Series(normed, index=dates)

def summarize_panel(df_paths, method="minmax", window=5):
    series_list = []
    n_failed = 0
    for _, row in df_paths.iterrows():
        fpath = row["file_y"]
        try:
            dates, ts, sign = load_timeseries(fpath)
            s = normalize_ts(dates, ts, sign, method=method)
            series_list.append(s.rename(row["ls_id"]))
        except Exception:
            n_failed += 1
            continue

    if len(series_list) == 0:
        return None, None, 0, 0, n_failed

    df_ts = pd.concat(series_list, axis=1)

    long = df_ts.stack().reset_index()
    long.columns = ["date", "ls_id", "value"]

    summary = long.groupby("date")["value"].agg(
        median=lambda x: np.nanmedian(x),
        p5=lambda x: np.nanpercentile(x, 5),
        p95=lambda x: np.nanpercentile(x, 95),
    ).sort_index()

    summary["med_sm"] = summary["median"].rolling(window, center=True, min_periods=1).mean()
    summary["p5_sm"]  = summary["p5"].rolling(window, center=True, min_periods=1).mean()
    summary["p95_sm"] = summary["p95"].rolling(window, center=True, min_periods=1).mean()

    n_series = df_ts.shape[1]
    n_points = (~df_ts.isna()).sum().sum()

    return long, summary, n_series, int(n_points), n_failed

def plot_panel(ax, long, summary, title, n_series, n_failed, norm_method):
    if long is None or summary is None:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return

    ax.grid(False)

    ax.axvspan(vel_t1, vel_t2, color=SPAN_WY22_COLOR, alpha=0.6, zorder=0)
    ax.axvspan(vel_t3, vel_t4, color=SPAN_WY23_COLOR, alpha=0.6, zorder=0)

    ax.scatter(long["date"], long["value"], color="lightgray", s=8, alpha=0.35, rasterized=True)
    ax.plot(summary.index, summary["median"], lw=1.2, linestyle="--", alpha=0.9, label="Median (raw)")
    ax.plot(summary.index, summary["med_sm"], lw=2.0, label="Median (smoothed)")
    ax.fill_between(summary.index, summary["p5_sm"], summary["p95_sm"], alpha=BAND_ALPHA, label="5–95% band")

    badge = f"N = {n_series}" + (f" (skipped {n_failed})" if n_failed else "")
    ax.text(0.02, 0.98, badge, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75))

    ax.set_title(title, pad=6, fontsize=11)
    ax.set_xlabel("Decimal Year")
    ax.set_ylabel("Normalized LOS displacement")

    if norm_method.lower() == "minmax":
        ax.set_ylim(-0.05, 1.05)

# =========================
# Load & prepare catalog
# =========================
df = pd.read_csv(CSV_PATH)

df["vel_dry1"] = np.abs(df["ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["ts_dry2_vel_myr"] * 100)
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

df = df[(df["vel_dry1"] >= vel_min_threshold) | (df["vel_dry2"] >= vel_min_threshold)].copy()

# PGA ratio and groups
df["pga_ratio"] = df["wy23_vs_wy22_pga_ratio"]

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

# =========================
# Map compiled CSV rows to HDF5 files
# =========================
h5_files = glob.glob(os.path.join(H5_DIR, "*.h5"))
h5_records = []
for fp in h5_files:
    try:
        with h5py.File(fp, "r") as hf:
            sid = hf["meta"].attrs.get("ID")
            h5_records.append({"ls_id": str(sid), "file_y": fp})
    except Exception:
        continue
df_h5 = pd.DataFrame(h5_records).dropna()

id_candidates = [c for c in df.columns if re.search(r"(ls_id|meta__ls_id|ID)$", c)]
if not id_candidates:
    raise RuntimeError("Could not find a landslide ID column in the CSV; set csv_id_col manually.")
csv_id_col = id_candidates[0]
df[csv_id_col] = df[csv_id_col].astype(str)

merged = df.merge(df_h5, left_on=csv_id_col, right_on="ls_id", how="left")

if merged["file_y"].isna().any():
    print(f"[info] {merged['file'].isna().sum()} slides lacked a matching HDF5 and will be skipped.")

# =========================
# Assemble 2×3 grid (by PGA groups)
# =========================
panel_groups = [
    ("Much Higher", True),
    ("Higher", True),
    ("Similar", True),
    ("Lower", True),
    ("Much Lower", True),
    ("Active (exclude Much Lower)", False)
]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12),
                         sharex=False, sharey=(NORM_METHOD=="minmax"))
axes = axes.flatten()

for ax, (name, is_group) in zip(axes, panel_groups):
    if is_group:
        sub = merged[(merged["group"] == name) & merged["file_y"].notna()][["ls_id","file_y"]].copy()
        title = f"{name}"
    else:
        sub = merged[(merged["group"] != "Much Lower") & merged["file_y"].notna()][["ls_id","file_y"]].copy()
        title = "Active (exclude Much Lower)"

    long, summary, n_series, n_points, n_failed = summarize_panel(sub, method=NORM_METHOD, window=SMOOTH_WIN)
    plot_panel(ax, long, summary, title, n_series, n_failed, NORM_METHOD)

# Labels (a–f)
for ax, lab in zip(axes, ['a','b','c','d','e','f']):
    ax.text(0.02, 1.065, f'{lab})', transform=ax.transAxes,
            ha='left', va='top', fontsize=12)

legend_handles = [
    Line2D([0],[0], color='C0', lw=2, label='Median (smoothed)'),
    Line2D([0],[0], color='C0', lw=1.2, label='Median (raw)'),
    Patch(facecolor='C0', alpha=BAND_ALPHA, label='5–95% band')
]
axes[-1].legend(handles=legend_handles, loc="lower right", frameon=True)

fig.suptitle(
    f"Normalized LOS time series by PGA-ratio group "
    f"(min≥{vel_min_threshold} cm/yr; norm={NORM_METHOD})",
    fontsize=14, y=0.995
)
plt.tight_layout(rect=[0, 0, 1, 0.98])

out_png = os.path.join(fig_dir, f"norm_ts_grid_byPGA_min{vel_min_threshold}_{NORM_METHOD}.png")
out_pdf = out_png.replace(".png", ".pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"→ saved: {out_png}")
print(f"→ saved: {out_pdf}")
plt.close(fig)

print("Done.")
