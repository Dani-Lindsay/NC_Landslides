#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset analysis with uncertainty (FAST, cached reads):
- Select slides with velocity ratio = "Much Faster" AND PGA ratio = "Much Higher"
- Build stacked normalized displacement median (plus 5–95% band)
- Also plot group-median cumulative 14d precipitation & PGA (normalized)
- Fit: pre-linear (k) before t1; post-exponential y = C - A exp(-beta (t - t1)) after t1
- Grid search over t1 around the EQ date; LS for k,b; grid for beta,C with continuity enforced
- Slide-level bootstrap with cached series (no re-reading HDF5 each draw)
- Output: PNG/PDF figure with bootstrap band; CSV with point estimates + 95% CIs + turning-point test

Tunable speed knobs are at the top: BOOT_N, BETA_GRID, C_GRID, T1_SEARCH_WINDOW_YEARS.
"""

import os, re, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from datetime import datetime, timedelta
from NC_Landslides_paths import *  # provides fig_dir etc.


# ---------------- User paths ----------------
try:
    from NC_Landslides_paths import *
    FIG_DIR = fig_dir
except Exception:
    FIG_DIR = "./figures"
    os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data_1/LS_Final_TS_4/compiled_landslide_data.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data_1/LS_Final_TS_4"

# ---------------- Parameters ----------------
vel_min_threshold = 2        # cm/yr filter for active slides
vel_multiple      = 5        # velocity-ratio bounds
NORM_METHOD       = "minmax"
SMOOTH_WIN        = 5        # display smoothing only

# Earthquake dates (decimal years)
EQ1     = 2021.9685          # 20 Dec 2021
EQ_DATE = 2022.9685          # 20 Dec 2022

# ---- Speed / stability knobs ----
BOOT_N                 = 1            # bootstrap draws (set to 1000+ for final numbers)
CONF_ALPHA             = 0.05           # 95% CI
PRE_WINDOW_YEARS       = 12/12          # ~4 months before t1 for pre-linear fit
T1_SEARCH_WINDOW_YEARS = 3/12           # search t1 within ±3 months of EQ
BETA_GRID              = np.linspace(0.1, 2.0, 100)  # /yr
C_GRID                 = np.linspace(0.7, 1.0, 40)   # normalized level

# ---------------- IO helpers ----------------
def load_disp(fpath):
    with h5py.File(fpath, "r") as hf:
        d = hf["dates"][:]
        ts = hf["clean_ts"][:]
        s  = hf["meta"].attrs.get("ls_sign", 1.0)
    return d, ts, s

def load_drivers(fpath):
    with h5py.File(fpath, "r") as hf:
        if "dates_14day_decimal" not in hf:
            return None, None, None
        d14 = hf["dates_14day_decimal"][:]
        precip = hf["/rainfall/rain_14day_cum"][:] if "/rainfall/rain_14day_cum" in hf else None
        pga    = hf["pga/pga_14day_cum"][:] if "pga/pga_14day_cum" in hf else None
    return d14, precip, pga

def normalize(vals, sign=1.0, method="minmax"):
    x = (vals - np.nanmean(vals)) * sign
    if method == "minmax":
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    elif method == "zscore":
        sd = np.nanstd(x)
        return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)
    else:
        raise ValueError("Unknown normalization")

# ---------------- Subset selection ----------------
def subset_muchFaster_muchHigher():
    df = pd.read_csv(CSV_PATH)
    df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
    df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)
    df = df[(df["vel_dry1"] >= vel_min_threshold) | (df["vel_dry2"] >= vel_min_threshold)].copy()
    df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

    def vel_label(r):
        if r < 1/vel_multiple: return "Much Slower"
        elif 1/vel_multiple <= r < 0.83: return "Slower"
        elif 0.83 <= r <= 1.2: return "Similar"
        elif 1.2 < r <= vel_multiple: return "Faster"
        else: return "Much Faster"
    df["vel_group"] = df["vel_ratio"].apply(vel_label)

    df["pga_ratio"] = df["support_params/wy23_vs_wy22_pga_ratio"]
    def pga_label(r):
        if r < 0.6: return "Much Lower"
        elif 0.6 <= r < 0.83: return "Lower"
        elif 0.83 <= r <= 1.2: return "Similar"
        elif 1.2 < r <= 3: return "Higher"
        else: return "Much Higher"
    df["pga_group"] = df["pga_ratio"].apply(pga_label)

    h5_files = glob.glob(os.path.join(H5_DIR, "*.h5"))
    recs = []
    for fp in h5_files:
        try:
            with h5py.File(fp, "r") as hf:
                sid = str(hf["meta"].attrs.get("ID"))
                recs.append({"ls_id": sid, "file": fp})
        except Exception:
            pass
    df_h5 = pd.DataFrame(recs)

    id_col = None
    for c in df.columns:
        if re.search(r"(ls_id|meta__ls_id|ID)$", c):
            id_col = c; break
    if id_col is None:
        raise RuntimeError("No landslide ID column in CSV")
    df[id_col] = df[id_col].astype(str)

    merged = df.merge(df_h5, left_on=id_col, right_on="ls_id", how="left")
    sub = merged[
        (merged["vel_group"] == "Much Faster") &
        (merged["pga_group"] == "Much Higher") &
        merged["file"].notna()
    ][["ls_id", "file"]].copy()
    return sub

# ---------------- Cache & stacking ----------------
def build_series_cache(sub):
    disp_cache, rain_series, pga_series = {}, [], []
    for _, r in sub.iterrows():
        try:
            d, ts, s = load_disp(r["file"])
            disp_cache[r["ls_id"]] = pd.Series(normalize(ts, s, NORM_METHOD), index=d, name=r["ls_id"])
            d14, precip, pga = load_drivers(r["file"])
            if d14 is not None and precip is not None:
                rain_series.append(pd.Series(normalize(precip, 1.0, NORM_METHOD), index=d14))
            if d14 is not None and pga is not None:
                pga_series.append(pd.Series(normalize(pga, 1.0, NORM_METHOD), index=d14))
        except Exception:
            continue
    rain_med = pd.concat(rain_series, axis=1).median(axis=1).sort_index() if rain_series else None
    pga_med  = pd.concat(pga_series, axis=1).median(axis=1).sort_index() if pga_series else None
    if not disp_cache:
        raise RuntimeError("Empty displacement cache")
    return disp_cache, rain_med, pga_med

def stack_from_cache(disp_cache, ids, smooth_win=SMOOTH_WIN):
    series = [disp_cache[i] for i in ids if i in disp_cache]
    df_disp = pd.concat(series, axis=1)
    long = df_disp.stack().reset_index()
    long.columns = ["date", "ls_id", "value"]
    summary = long.groupby("date")["value"].agg(
        median=lambda x: np.nanmedian(x),
        p5=lambda x: np.nanpercentile(x, 5),
        p95=lambda x: np.nanpercentile(x, 95),
    ).sort_index()
    summary["med_sm"] = summary["median"].rolling(smooth_win, center=True, min_periods=1).mean()
    summary["p5_sm"]  = summary["p5"].rolling(smooth_win, center=True, min_periods=1).mean()
    summary["p95_sm"] = summary["p95"].rolling(smooth_win, center=True, min_periods=1).mean()
    return long, summary

# ---------------- Fit model ----------------
def fit_prelinear_postexp(d, y, eq_date,
                          pre_window_years=PRE_WINDOW_YEARS,
                          t1_window=T1_SEARCH_WINDOW_YEARS,
                          beta_grid=BETA_GRID, c_grid=C_GRID):
    d = np.asarray(d, float); y = np.asarray(y, float)
    mfin = np.isfinite(d) & np.isfinite(y)
    d, y = d[mfin], y[mfin]
    t1_candidates = d[(d >= eq_date - t1_window) & (d <= eq_date + t1_window)]
    if t1_candidates.size == 0: t1_candidates = np.array([eq_date])

    best = None
    for t1 in t1_candidates:
        mp = (d <= t1) & (d >= t1 - pre_window_years)
        tt, yy = (d[mp], y[mp]) if mp.sum() >= 2 else (d[np.argsort(np.abs(d-(t1-pre_window_years/2)))[:3]],
                                                      y[np.argsort(np.abs(d-(t1-pre_window_years/2)))[:3]])
        A_lin = np.vstack([tt, np.ones_like(tt)]).T
        k, b = np.linalg.lstsq(A_lin, yy, rcond=None)[0]
        mpost = d >= t1
        if mpost.sum() < 2: continue
        y_t1 = k*t1+b
        for C in c_grid:
            A0 = C-y_t1
            for beta in beta_grid:
                yhat_full = np.where(d<t1, k*d+b, C-A0*np.exp(-beta*(d-t1)))
                rmse = np.sqrt(np.nanmean((yhat_full-y)**2))
                if (best is None) or (rmse<best["rmse"]):
                    best = dict(k=k,b=b,C=C,beta=beta,t1=t1,rmse=rmse,yhat=yhat_full)
    if best is None: raise RuntimeError("Fit failed")
    beta,k,b,t1,C = best["beta"],best["k"],best["b"],best["t1"],best["C"]
    y_t1=k*t1+b; A0=C-y_t1
    t50 = math.log(2.0)/beta if beta>0 else np.inf
    t90 = math.log(10.0)/beta if beta>0 else np.inf
    tau_star = np.log(max(A0*beta,1e-12)/max(k,1e-12))/beta if (beta>0 and k>0 and A0>0) else np.nan
    best.update(dict(t50=t50,t90=t90,tau_star=tau_star,A0=A0,y_t1=y_t1))
    return best

# ---------------- Bootstrap ----------------
def bootstrap_params_cached(sub, eq_date, B=BOOT_N, seed=42):
    rng = np.random.default_rng(seed)
    ids_all = sub["ls_id"].unique().tolist()
    disp_cache, rain_med, pga_med = build_series_cache(sub)
    _, base_summary = stack_from_cache(disp_cache, ids_all)
    d_grid = base_summary.index.values.astype(float)
    rows, all_yhat = [], []
    for b in range(B):
        boot_ids = rng.choice(ids_all, size=len(ids_all), replace=True)
        _, ssum = stack_from_cache(disp_cache, boot_ids)
        d, y = ssum.index.values.astype(float), ssum["median"].values
        fit = fit_prelinear_postexp(d,y,eq_date)
        rows.append([fit["k"],fit["beta"],fit["t50"],fit["t90"],fit["tau_star"],fit["t1"],fit["rmse"]])
        all_yhat.append(np.interp(d_grid,d,fit["yhat"]))
    boot = pd.DataFrame(rows,columns=["k","beta","t50","t90","tau_star","t1","rmse"])
    band=None
    if all_yhat:
        Y=np.vstack(all_yhat)
        lo=np.nanpercentile(Y,100*CONF_ALPHA/2,axis=0)
        hi=np.nanpercentile(Y,100*(1-CONF_ALPHA/2),axis=0)
        band=pd.DataFrame({"date":d_grid,"yhat_lo":lo,"yhat_hi":hi}).set_index("date")
    return boot,band,base_summary,rain_med,pga_med,disp_cache

# ---------------- Main ----------------
def main():
    sub=subset_muchFaster_muchHigher(); N=len(sub)
    print(f"Subset size (Much Faster ∩ Much Higher): N={N}")
    boot, band, summary, rain_med, pga_med, disp_cache = bootstrap_params_cached(sub, EQ_DATE, B=BOOT_N)
    d=summary.index.values.astype(float); y=summary["median"].values
    fit=fit_prelinear_postexp(d,y,EQ_DATE)
    print(f"Fit: k={fit['k']:.3f}/yr, beta={fit['beta']:.3f}/yr, t1={fit['t1']:.3f}, RMSE={fit['rmse']:.3f}")
    print(f"      t50={fit['t50']:.2f} yr, t90={fit['t90']:.2f} yr, tau*={fit['tau_star']:.2f} yr")

    # build long_disp for scatter
    long_disp = pd.concat(disp_cache.values(), axis=1).stack().reset_index()
    long_disp.columns = ["date", "ls_id", "value"]

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(7,5))
    
    ax.scatter(long_disp["date"], long_disp["value"], s=4, c="k", alpha=0.15,
               zorder=1, rasterized=True, edgecolors="none",)
    
    ax.axvline(EQ1,color="#444",lw=1.2,alpha=0.6); 
    ax.axvline(EQ_DATE,color="#444",lw=1.2,alpha=0.6, zorder=2)
    
    if rain_med is not None: ax.plot(rain_med.index,rain_med.values,color="lightseagreen",lw=1.2,ls="--",zorder=3,label="Median cum. precip")
    if pga_med is not None:  ax.plot(pga_med.index,pga_med.values,color="#d95f0e",lw=1.2,ls="--",zorder=4,label="Median cum. PGA")
    
    ax.fill_between(summary.index, summary["p5_sm"], summary["p95_sm"], color="#aec7e8", alpha=0.25, zorder=5, label="5–95% disp")
    if band is not None:
        ax.fill_between(band.index, band["yhat_lo"], band["yhat_hi"], color="#ffbb78", alpha=0.25, zorder=6, label="Model 95% band (bootstrap)")
    
    ax.plot(summary.index, np.interp(d,d,fit["yhat"]), color="#ff7f0e", lw=2.5, zorder=7, label="Model (pre linear, post exp)")
    ax.plot(summary.index, summary["median"], lw=1.2, color="#1f77b4", ls="--", zorder=9, label="Median Displacement")
    ax.plot(summary.index, summary["med_sm"], lw=2.0, color="#1f77b4", zorder=8, label="Median Smoothed")
    
    ax.set_ylabel("Normalized data"); ax.set_xlabel("Decimal Year"); ax.set_ylim(-0.02,1.02)
    ax.set_title('Subset: Velocity "Much Faster" ∩ PGA "Much Higher"')
    ax.legend(loc="upper left",facecolor='white'); plt.tight_layout()
    out_png=os.path.join(FIG_DIR,f"subset_muchFaster_muchHigher_withscatter_prequakewindow{PRE_WINDOW_YEARS}_nBoots{BOOT_N}.png")
    out_pdf=out_png.replace(".png",".pdf"); fig.savefig(out_png,dpi=300,bbox_inches="tight"); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)
    print(f"→ saved: {out_png}, {out_pdf}")
    
    # ---------------- CSV summary ----------------
    
    # CIs
    def ci(s):
        return np.nanpercentile(s, [100*CONF_ALPHA/2, 100*(1-CONF_ALPHA/2)]).tolist()
    
    k_ci    = ci(boot["k"])
    beta_ci = ci(boot["beta"])
    t50_ci  = ci(boot["t50"])
    t90_ci  = ci(boot["t90"])
    tau_ci  = ci(boot["tau_star"])
    t1_ci   = ci(boot["t1"])
    
    # Turning point test wrt EQ_DATE
    frac_after  = np.mean(boot["t1"] > EQ_DATE)
    frac_before = np.mean(boot["t1"] < EQ_DATE)
    eq_in_ci    = (t1_ci[0] <= EQ_DATE <= t1_ci[1])

    rows = [{
        "N": len(sub),
        "k": fit["k"], "k_lo": k_ci[0], "k_hi": k_ci[1],
        "beta": fit["beta"], "beta_lo": beta_ci[0], "beta_hi": beta_ci[1],
        "t50": fit["t50"], "t50_lo": t50_ci[0], "t50_hi": t50_ci[1],
        "t90": fit["t90"], "t90_lo": t90_ci[0], "t90_hi": t90_ci[1],
        "tau_star": fit["tau_star"], "tau_lo": tau_ci[0], "tau_hi": tau_ci[1],
        "t1_fit": fit["t1"], "t1_lo": t1_ci[0], "t1_hi": t1_ci[1],
        "eq_date": EQ_DATE,
        "P_t1_greater_EQ": float(frac_after),
        "P_t1_less_EQ": float(frac_before),
        "EQ_in_t1_CI": bool(eq_in_ci),
        "rmse": fit["rmse"],
        "BOOT_N": BOOT_N,
        "BETA_MIN": float(BETA_GRID.min()), "BETA_MAX": float(BETA_GRID.max()), "BETA_N": len(BETA_GRID),
        "C_MIN": float(C_GRID.min()), "C_MAX": float(C_GRID.max()), "C_N": len(C_GRID),
        "T1_WINDOW_YEARS": T1_SEARCH_WINDOW_YEARS,
        "PRE_WINDOW_YEARS": PRE_WINDOW_YEARS,
    }]
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(
        FIG_DIR,
        f"subset_muchFaster_muchHigher_fit_summary_prequakewindow{PRE_WINDOW_YEARS}_nBoots{BOOT_N}.csv"
    )
    df_out.to_csv(out_csv, index=False)
    print(f"→ saved: {out_csv}")

if __name__=="__main__":
    main()
