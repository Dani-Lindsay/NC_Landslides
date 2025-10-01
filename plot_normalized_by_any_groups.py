#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalized LOS displacement + cumulative precipitation + cumulative PGA
by multiple grouping strategies (velocity, PGA ratio, precipitation ratio).
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
BAND_ALPHA  = 0.2

# Earthquake dates (decimal years)
eq1 = 2021.9685   # 20 Dec 2021
eq2 = 2022.9685   # 20 Dec 2022

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

    # Precipitation (blue line)
    if summary_rain is not None:
        ax.plot(summary_rain.index, summary_rain["median"],
                lw=1.8, color="royalblue", zorder=4)

    # PGA (purple line)
    if summary_pga is not None:
        ax.plot(summary_pga.index, summary_pga["median"],
                lw=1.8, color="purple", zorder=5)
        
    # Median displacement (dashed black)
    ax.plot(summary_disp.index, summary_disp["median"],
            lw=1.2, color="black", linestyle="--", zorder=6)

    # Smoothed displacement (highlight, burnt orange)
    ax.plot(summary_disp.index, summary_disp["med_sm"],
            lw=2.2, color="darkorange", zorder=7)

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
    df = df.copy()
    df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]
    def categorize(ratio):
        if ratio < 1 / vel_multiple: return "Much Slower (Vel 2023 / Vel 2022)"
        elif 1 / vel_multiple <= ratio < 0.83: return "Slower (Vel 2023 / Vel 2022)"
        elif 0.83 <= ratio <= 1.2: return "Similar (Vel 2023 / Vel 2022)"
        elif 1.2 < ratio <= vel_multiple: return "Faster (Vel 2023 / Vel 2022)"
        else: return "Much Faster (Vel 2023 / Vel 2022)"
    df["group"] = df["vel_ratio"].apply(categorize)
    return df

def group_by_pga_ratio(df):
    df = df.copy()
    df["pga_ratio"] = df["support_params/wy23_vs_wy22_pga_ratio"]
    def categorize(ratio):
        if ratio < 0.6: return "Much Lower (PGA 2023 / PGA 2022)"
        elif 0.6 <= ratio < 0.83: return "Lower (PGA 2023 / PGA 2022)"
        elif 0.83 <= ratio <= 1.2: return "Similar (PGA 2023 / PGA 2022)"
        elif 1.2 < ratio <= 3: return "Higher (PGA 2023 / PGA 2022)"
        else: return "Much Higher (PGA 2023 / PGA 2022)"
    df["group"] = df["pga_ratio"].apply(categorize)
    return df

def group_by_precip_ratio(df):
    df = df.copy()
    df["precip_ratio"] = df["support_params/wy23_vs_wy22_rain_ratio"]
    def categorize(ratio):
        if ratio <= 1.1: return "Similar (Rainfall 2023 / Rainfall 2022)"
        elif 1.1 < ratio <= 1.5: return "Higher (Rainfall 2023 / Rainfall 2022)"
        else: return "Much Higher (Rainfall 2023 / Rainfall 2022)"
    df["group"] = df["precip_ratio"].apply(categorize)
    return df

def group_by_area(df):
    """Categorize landslide area into 6 log10-spaced bins with interpretable labels."""
    df = df.copy()

    def categorize(area_m2):
        if pd.isna(area_m2) or area_m2 <= 0:
            return "Unknown"

        log_area = np.log10(area_m2)
        if log_area < 4.0:            
            return "Smallest (<1e4 m²)"
        elif 4.0 <= log_area < 4.5:   
            return "Small (1e4–3e4 m²)"
        elif 4.5 <= log_area < 5.0:   
            return "Medium-Small (3e4–1e5 m²)"
        elif 5.0 <= log_area < 5.5:   
            return "Medium (1e5–3e5 m²)"
        elif 5.5 <= log_area < 6.0:   
            return "Large (3e5–1e6 m²)"
        else:                         
            return "Largest (≥1e6 m²)"

    df["group"] = df["meta__ls_area_m2"].apply(categorize)
    return df

# -------------------------
# Dictionary of available strategies
# -------------------------
GROUPING_STRATEGIES = {
    "velocity": group_by_velocity_ratio,
    "pga": group_by_pga_ratio,
    "precip": group_by_precip_ratio,
    "area": group_by_area,
}

# -------------------------
# Main plotting routine
# -------------------------
def run_panels(df, grouping_name="custom"):
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

    # Merge
    id_candidates = [c for c in df.columns if re.search(r"(ls_id|meta__ls_id|ID)$", c)]
    if not id_candidates:
        raise RuntimeError("Could not find a landslide ID column in the CSV")
    csv_id_col = id_candidates[0]
    df[csv_id_col] = df[csv_id_col].astype(str)
    merged = df.merge(df_h5, left_on=csv_id_col, right_on="ls_id", how="left")

    # Panel groups
    panel_groups = [(g, True) for g in sorted(merged["group"].unique())]

    fig, axes = plt.subplots(nrows=(len(panel_groups)+1)//2, ncols=2,
                             figsize=(12, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    results = []
    for ax, (name, _) in zip(axes, panel_groups):
        sub = merged[(merged["group"] == name) & merged["file"].notna()][["ls_id","file"]].copy()
        long_disp, summary_disp, summary_rain, summary_pga, n_series, n_points, n_failed = summarize_panel(
            sub, method=NORM_METHOD, window=SMOOTH_WIN
        )
        plot_panel(ax, long_disp, summary_disp, summary_rain, summary_pga, name, n_series, n_failed)
        results.append({"group": name, "N": n_series, "n_failed": n_failed})

    for ax, lab in zip(axes, list("abcdefghijklmnopqrstuvwxyz")[:len(panel_groups)]):
        ax.text(0.02, 1.065, f'{lab})', transform=ax.transAxes, ha='left', va='top', fontsize=12)

    legend_handles = [
        Line2D([0],[0], lw=2.2, color="darkorange", label='Smoothed disp.'),
        Line2D([0],[0], lw=1.2, color="black", linestyle="--", label='Median disp'),
        Patch(facecolor="steelblue", alpha=BAND_ALPHA, label='5–95% disp'),
        Line2D([0],[0], lw=1.8, color="royalblue", label='Median cum. precip.'),
        Line2D([0],[0], lw=1.8, color="purple", label='Median cum. PGA'),
        Line2D([0],[0], color="red", alpha=0.5, linestyle="--", lw=1.5, label='Earthquakes'),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.suptitle(
        f"Normalized LOS displacement + 14-day cumulative precipitation + PGA grouped by {grouping_name}",
        fontsize=14, y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = os.path.join(fig_dir, f"norm_ts_{grouping_name}_groups.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(fig_dir, f"norm_ts_{grouping_name}_groups.csv")
    results_df.to_csv(out_csv, index=False)

    print(f"→ saved: {out_png}, {out_pdf}, {out_csv}")

# -------------------------
# Run all groupings
# -------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
    df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
    df = df[(df["vel_dry1"] >= vel_min_threshold) | (df["vel_dry2"] >= vel_min_threshold)].copy()
    df["meta__ls_area_m2_log10"] = np.log10(df["meta__ls_area_m2"])
    
    for name, func in GROUPING_STRATEGIES.items():
        print(f"\n=== Running grouping: {name} ===")
        df_grouped = func(df)
        run_panels(df_grouped, grouping_name=name)

# -------------------------
# Median comparison plot
# -------------------------
def plot_group_median_comparison(group_summaries, grouping_name="custom"):
    """
    Create a single figure showing smoothed median displacement curves
    for all groups from the given grouping strategy.
    """
    if not group_summaries:
        print(f"[WARN] No group summaries to plot for {grouping_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use a color palette to differentiate groups
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_summaries)))

    for (name, summary), color in zip(group_summaries.items(), colors):
        ax.plot(summary.index, summary["med_sm"], lw=2.0, label=name, color=color)

    # Add earthquake markers
    ax.axvline(eq1, color="red", linestyle="--", lw=1.5, zorder=1, label="EQ 2021")
    ax.axvline(eq2, color="red", linestyle="--", lw=1.5, zorder=1, label="EQ 2022")

    ax.set_ylabel("Normalized displacement")
    ax.set_xlabel("Decimal Year")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Group comparison: smoothed medians ({grouping_name})")

    # One legend for everything
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()

    out_summary = os.path.join(fig_dir, f"norm_ts_{grouping_name}_median_comparison.png")
    out_summary_pdf = out_summary.replace(".png", ".pdf")
    fig.savefig(out_summary, dpi=300, bbox_inches="tight")
    fig.savefig(out_summary_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"→ saved comparison: {out_summary}, {out_summary_pdf}")


# -------------------------
# Update run_panels to collect summaries
# -------------------------
def run_panels(df, grouping_name="custom"):
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

    # Merge
    id_candidates = [c for c in df.columns if re.search(r"(ls_id|meta__ls_id|ID)$", c)]
    if not id_candidates:
        raise RuntimeError("Could not find a landslide ID column in the CSV")
    csv_id_col = id_candidates[0]
    df[csv_id_col] = df[csv_id_col].astype(str)
    merged = df.merge(df_h5, left_on=csv_id_col, right_on="ls_id", how="left")

    # Panel groups
    panel_groups = [(g, True) for g in sorted(merged["group"].unique())]

    fig, axes = plt.subplots(nrows=(len(panel_groups)+1)//2, ncols=2,
                             figsize=(12, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    results = []
    group_summaries = {}  # collect median summaries here
    for ax, (name, _) in zip(axes, panel_groups):
        sub = merged[(merged["group"] == name) & merged["file"].notna()][["ls_id","file"]].copy()
        long_disp, summary_disp, summary_rain, summary_pga, n_series, n_points, n_failed = summarize_panel(
            sub, method=NORM_METHOD, window=SMOOTH_WIN
        )
        plot_panel(ax, long_disp, summary_disp, summary_rain, summary_pga, name, n_series, n_failed)
        results.append({"group": name, "N": n_series, "n_failed": n_failed})
        if summary_disp is not None:
            group_summaries[name] = summary_disp.copy()

    for ax, lab in zip(axes, list("abcdefghijklmnopqrstuvwxyz")[:len(panel_groups)]):
        ax.text(0.02, 1.065, f'{lab})', transform=ax.transAxes, ha='left', va='top', fontsize=12)

    legend_handles = [
        Line2D([0],[0], lw=2.2, color="darkorange", label='Smoothed disp.'),
        Line2D([0],[0], lw=1.2, color="black", linestyle="--", label='Median disp'),
        Patch(facecolor="steelblue", alpha=BAND_ALPHA, label='5–95% disp'),
        Line2D([0],[0], lw=1.8, color="royalblue", label='Median cum. precip.'),
        Line2D([0],[0], lw=1.8, color="purple", label='Median cum. PGA'),
        Line2D([0],[0], color="red", alpha=0.5, linestyle="--", lw=1.5, label='Earthquakes'),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.suptitle(
        f"Normalized LOS displacement + 14-day cumulative precipitation + PGA grouped by {grouping_name}",
        fontsize=14, y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = os.path.join(fig_dir, f"norm_ts_{grouping_name}_groups.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(fig_dir, f"norm_ts_{grouping_name}_groups.csv")
    results_df.to_csv(out_csv, index=False)

    print(f"→ saved: {out_png}, {out_pdf}, {out_csv}")

    # NEW: plot all group medians in one comparison subplot
    plot_group_median_comparison(group_summaries, grouping_name)

