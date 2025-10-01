#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset plot with pre-linear + post-exponential fit
Filter: velocity group ∩ PGA group (user-set)
Displacement: scatter, 5–95% band, median, smoothed median
Drivers: group-median cumulative 14-day rainfall and PGA
Fit: y = k t + b (pre); y = C - (C - y1) exp(-β (t - t1)) (post), continuous at t1
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import h5py

# ----------------- user paths -----------------
from NC_Landslides_paths import *  # provides fig_dir or similar

CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"

# ----------------- parameters -----------------
vel_min_threshold = 2          # cm/yr
vel_multiple      = 5          # for ratio grouping
NORM_METHOD       = "minmax"
SMOOTH_WIN        = 5
BAND_ALPHA        = 0.25

# Earthquake dates (decimal years)
EQ1 = 2021.9685     # 20 Dec 2021
EQ2 = 2022.9685     # 20 Dec 2022  (fit anchor t1)
T1  = EQ2

# Choose subset here:
TARGET_VEL = "Much Faster"     # or "Much Slower", "Faster", "Similar", "Slower"
TARGET_PGA = "Much Higher"     # "Much Higher", "Higher", "Similar", "Lower", "Much Lower"

# ----------------- helpers: I/O & normalization -----------------
def load_timeseries(fpath):
    with h5py.File(fpath, "r") as hf:
        dates = hf["dates"][:]
        ts    = hf["clean_ts"][:]
        sign  = hf["meta"].attrs.get("ls_sign", 1.0)
    return dates, ts, float(sign)

def load_drivers(fpath):
    """Return (date_14d, cum_rain, cum_pga) or (None,...)."""
    with h5py.File(fpath, "r") as hf:
        if "dates_14day_decimal" not in hf:
            return None, None, None
        d14 = hf["dates_14day_decimal"][:]
        rain = hf["/rainfall/rain_14day_cum"][:] if "/rainfall/rain_14day_cum" in hf else None
        pga  = hf["pga/pga_14day_cum"][:]      if "pga/pga_14day_cum" in hf      else None
    return d14, rain, pga

def normalize_series(vals, sign=1.0, method="minmax"):
    x = (np.asarray(vals, float) - np.nanmean(vals)) * sign
    if method == "minmax":
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo)/(hi - lo) if hi > lo else np.zeros_like(x)
    elif method == "zscore":
        s = np.nanstd(x)
        return (x - np.nanmean(x))/s if s > 0 else np.zeros_like(x)
    else:
        raise ValueError("Unknown normalization")

# ----------------- grouping rules (match your earlier thresholds) -----------------
def add_velocity_group(df, vel_multiple=5):
    df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
    df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
    df = df[(df["vel_dry1"] >= vel_min_threshold) | (df["vel_dry2"] >= vel_min_threshold)].copy()
    df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]
    def cat(r):
        if r < 1/vel_multiple:        return "Much Slower"
        elif 1/vel_multiple <= r < .83:  return "Slower"
        elif .83 <= r <= 1.2:         return "Similar"
        elif 1.2 < r <= vel_multiple: return "Faster"
        else:                         return "Much Faster"
    df["vel_group"] = df["vel_ratio"].apply(cat)
    return df

def add_pga_group(df):
    df["pga_ratio"] = df["support_params/wy23_vs_wy22_pga_ratio"]
    def cat(r):
        if r < 0.6:          return "Much Lower"
        elif r < 0.83:       return "Lower"
        elif r <= 1.2:       return "Similar"
        elif r <= 3.0:       return "Higher"
        else:                return "Much Higher"
    df["pga_group"] = df["pga_ratio"].apply(cat)
    return df

# ----------------- summary for a set of H5 files -----------------
def summarize_subset(id_file_df, norm_method="minmax", smooth_win=5):
    disp_series, rain_series, pga_series = [], [], []

    for _, row in id_file_df.iterrows():
        fpath = row["file"]
        try:
            d, ts, sgn = load_timeseries(fpath)
            disp_series.append(pd.Series(normalize_series(ts, sgn, norm_method), index=d, name=row["ls_id"]))
            d14, rain, pga = load_drivers(fpath)
            if d14 is not None and rain is not None:
                rain_series.append(pd.Series(normalize_series(rain, 1.0, norm_method), index=d14, name=row["ls_id"]))
            if d14 is not None and pga is not None:
                pga_series.append(pd.Series(normalize_series(pga, 1.0, norm_method), index=d14, name=row["ls_id"]))
        except Exception as e:
            print(f"[WARN] {os.path.basename(fpath)}: {e}")
            continue

    if not disp_series:
        return None

    # Displacement long form
    df_disp = pd.concat(disp_series, axis=1)
    long_disp = df_disp.stack().reset_index()
    long_disp.columns = ["date", "ls_id", "value"]

    # Percentiles & medians per date
    g = long_disp.groupby("date")["value"]
    summary_disp = pd.DataFrame({
        "median": g.median(),
        "p5":     g.quantile(0.05),
        "p95":    g.quantile(0.95),
    }).sort_index()
    summary_disp["med_sm"] = summary_disp["median"].rolling(smooth_win, center=True, min_periods=1).mean()
    summary_disp["p5_sm"]  = summary_disp["p5"].rolling(smooth_win, center=True, min_periods=1).mean()
    summary_disp["p95_sm"] = summary_disp["p95"].rolling(smooth_win, center=True, min_periods=1).mean()

    # Group-median drivers
    summary_rain = None
    if rain_series:
        df_r = pd.concat(rain_series, axis=1)
        summary_rain = pd.DataFrame({"median": df_r.median(axis=1)}).sort_index()

    summary_pga = None
    if pga_series:
        df_g = pd.concat(pga_series, axis=1)
        summary_pga = pd.DataFrame({"median": df_g.median(axis=1)}).sort_index()

    return {
        "long_disp": long_disp,
        "summary_disp": summary_disp,
        "summary_rain": summary_rain,
        "summary_pga": summary_pga,
        "N": df_disp.shape[1]
    }

# ----------------- piecewise fit (continuous at t1) -----------------
def fit_prelinear_postexp(d, y, t1, pre_window_years=4/12, c_grid=np.linspace(0.6, 1.2, 61)):
    """
    Fit y(t)=k t + b for t<t1;
        y(t)=C - (C - y1) * exp(-β (t - t1)) for t>=t1, continuous at t1.
    Returns dict: {"k","b","beta","C","yhat","rmse"}.
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)

    # --- pre-linear (use last 'pre_window_years' before t1)
    pre_mask = d < t1
    if pre_mask.sum() < 2:
        k = 0.0
        b = float(y[pre_mask][-1]) if pre_mask.any() else float(y[~pre_mask][0])
    else:
        pre_d = d[pre_mask]
        use = pre_d >= (t1 - pre_window_years)
        X = np.vstack([pre_d[use], np.ones(use.sum())]).T
        k, b = np.linalg.lstsq(X, y[pre_mask][use], rcond=None)[0]

    y1 = k*t1 + b

    # --- post-exp β via ln regression for each C in grid
    post_mask = ~pre_mask
    tpost = d[post_mask] - t1
    ypost = y[post_mask]

    best = None
    for C in c_grid:
        s = C - y1
        if s <= 0:
            continue

        z = C - ypost
        valid = z > 0
        if valid.sum() < 2:
            continue

        lnz = np.log(z[valid])
        X = np.vstack([np.ones(valid.sum()), -tpost[valid]]).T  # ln z = ln s - β t
        alpha, beta = np.linalg.lstsq(X, lnz, rcond=None)[0]   # alpha≈ln s

        # predictions
        zhat_all = np.exp(alpha - beta * tpost)                # for all post points
        yhat_post = C - zhat_all

        yhat = k*d + b                                         # full vector
        yhat[post_mask] = yhat_post                            # fill post part

        rmse = np.sqrt(np.nanmean((yhat - y)**2))
        if (best is None) or (rmse < best["rmse"]):
            best = {"k": k, "b": b, "beta": beta, "C": C, "yhat": yhat, "rmse": rmse}

    # fallback if grid fails
    if best is None:
        yhat = k*d + b
        best = {"k": k, "b": b, "beta": 0.0, "C": y1 + 1e-6, "yhat": yhat,
                "rmse": np.sqrt(np.nanmean((yhat - y)**2))}
    return best

# ----------------- main -----------------
if __name__ == "__main__":
    # --- load master CSV and build groups ---
    df = pd.read_csv(CSV_PATH)
    df = add_velocity_group(df, vel_multiple=vel_multiple)
    df = add_pga_group(df)

    # map H5 files to ls_id
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

    # choose subset
    id_col = next(c for c in df.columns if re.search(r"(ls_id|meta__ls_id|ID)$", c))
    df[id_col] = df[id_col].astype(str)

    subset = df[(df["vel_group"] == TARGET_VEL) & (df["pga_group"] == TARGET_PGA)].copy()
    merged = subset.merge(df_h5, left_on=id_col, right_on="ls_id", how="left")
    sub_files = merged[merged["file"].notna()][["ls_id","file"]].copy()

    if sub_files.empty:
        raise SystemExit(f"No timeseries found for subset: vel={TARGET_VEL}, pga={TARGET_PGA}")

    # --- summarize displacement & drivers ---
    S = summarize_subset(sub_files, norm_method=NORM_METHOD, smooth_win=SMOOTH_WIN)
    long_disp   = S["long_disp"]
    summary     = S["summary_disp"]
    summary_r   = S["summary_rain"]
    summary_g   = S["summary_pga"]
    Nseries     = S["N"]

    # --- fit piecewise model to smoothed median ---
    d = summary.index.values.astype(float)
    y = summary["med_sm"].values
    fit = fit_prelinear_postexp(d, y, t1=T1, pre_window_years=4/12)
    k, beta, C, rmse = fit["k"], fit["beta"], fit["C"], fit["rmse"]
    t12 = np.log(2.0)/beta if beta > 0 else np.inf
    t90 = np.log(10.0)/beta if beta > 0 else np.inf

    # --- plot ---
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 6))

    # EQ lines
    ax.axvline(EQ1, color="red", linestyle="--", alpha=0.6, lw=1.4)
    ax.axvline(EQ2, color="red", linestyle="--", alpha=0.6, lw=1.4)

    # scatter + band
    ax.scatter(long_disp["date"], long_disp["value"], s=6, c="k", alpha=0.15, zorder=1, rasterized=True)
    ax.fill_between(summary.index, summary["p5_sm"], summary["p95_sm"],
                    color="steelblue", alpha=BAND_ALPHA, zorder=2, label="5–95% disp")

    # drivers
    if summary_r is not None:
        ax.plot(summary_r.index, summary_r["median"], color="#6C8AE4", lw=1.8, label="Median cum. precip")
    if summary_g is not None:
        ax.plot(summary_g.index, summary_g["median"], color="#8B4A52", lw=1.8, label="Median cum. PGA")

    # medians
    ax.plot(summary.index, summary["median"],  color="k", lw=1.2, ls="--", label="Median disp")
    ax.plot(summary.index, summary["med_sm"], color="darkorange", lw=2.2, label="Smoothed disp")

    # fitted curve
    ax.plot(d, fit["yhat"], color="green", lw=2.4, label="Piecewise fit (pre linear, post exp)")

    ax.set_ylabel("Normalized data")
    ax.set_xlabel("Decimal Year")
    ax.set_ylim(-0.02, 1.02)
    title = f'Subset: Velocity "{TARGET_VEL}" ∩ PGA "{TARGET_PGA}"'
    ax.set_title(title)

    # badge with fit stats
    txt = (f"Pre slope k = {k:.3f} /yr\n"
           f"β = {beta:.3f} /yr  →  t½={t12:.2f} yr,  t90={t90:.2f} yr\n"
           f"RMSE = {rmse:.3f}   (N={Nseries})")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="k", alpha=0.9))

    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_png = os.path.join(fig_dir, f"subset_{TARGET_VEL.replace(' ','')}_{TARGET_PGA.replace(' ','')}_with_fit.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"→ saved: {out_png}, {out_pdf}")
