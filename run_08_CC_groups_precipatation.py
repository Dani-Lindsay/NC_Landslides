#!/usr/bin/env python3
"""
Cross-correlation on group-level (median) displacement vs cumulative rainfall.
- Groups defined by velocity ratio (Much Faster, Faster, Similar, Slower, Much Slower, Active).
- For each group: median normalized displacement + cumulative rainfall -> cross-correlation.
- Saves per-group figures and one CSV with lag results.
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.signal import correlate, correlation_lags

# =========================
# Config
# =========================
CSV_PATH = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv"
H5_DIR   = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"
OUT_DIR  = "/Volumes/Seagate/NC_Landslides/Figures/GroupCorrelation"
os.makedirs(OUT_DIR, exist_ok=True)

VEL_MIN_THRESHOLD = 2     # cm/yr threshold
VEL_MULTIPLE      = 5
MAX_LAG_DAYS      = 120

# =========================
# Helpers
# =========================
def decimal_year_to_datetime(dyear):
    year = int(dyear)
    rem = dyear - year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year+1, month=1, day=1)
    return start + (end - start) * rem

def load_timeseries(fpath):
    with h5py.File(fpath, "r") as hf:
        dates = [decimal_year_to_datetime(d) for d in hf["dates"][:]]
        ts = hf["clean_ts"][:]
        sign = hf["meta"].attrs.get("ls_sign", 1.0)
        rain_dates = pd.to_datetime(hf["rainfall/date"][:].astype(str)).tz_localize(None)
        rain_vals = hf["rainfall/rain_mm"][:]
    return np.array(dates), ts * sign, pd.Series(rain_vals, index=rain_dates)

def normalize_minmax(arr):
    arr = np.array(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

def categorize_ratio_group(ratio, multiple=VEL_MULTIPLE):
    if ratio < 1/multiple: return "Much Slower"
    elif ratio < 0.83: return "Slower"
    elif ratio <= 1.2: return "Similar"
    elif ratio <= multiple: return "Faster"
    else: return "Much Faster"

def find_custom_splits(dates, rain):
    """Define Period1 and Period2 using rainfall with fallbacks."""
    dates = pd.to_datetime(dates)
    rain = np.array(rain)

    # Period 1: start -> last zero in 2022
    mask_2022 = (pd.Series(dates).dt.year == 2022).values
    idx_zero = np.where((rain == 0) & mask_2022)[0]
    end1 = dates[idx_zero[-1]] if len(idx_zero)>0 else pd.Timestamp("2022-07-01")
    start1 = dates[0]

    # Period 2: first dry mid-2022 -> first rain in autumn 2023
    mask_mid2022 = (dates > pd.Timestamp("2022-06-01")) & (dates < pd.Timestamp("2022-12-01"))
    idx_zero = np.where((rain == 0) & mask_mid2022)[0]
    if len(idx_zero) > 1 and np.any(np.diff(idx_zero) == 1):
        start2 = dates[idx_zero[np.where(np.diff(idx_zero)==1)[0][0]]]
    else:
        start2 = pd.Timestamp("2022-07-01")
    mask_autumn2023 = (dates > pd.Timestamp("2023-09-01"))
    idx_rain = np.where((rain > 0) & mask_autumn2023)[0]
    end2 = dates[idx_rain[0]] if len(idx_rain)>0 else dates[-1]

    return (start1,end1),(start2,end2)

def cross_correlation(dates, disp, rain, start, end, max_lag_days=120):
    mask = (dates >= start) & (dates <= end)
    if mask.sum() < 5:
        return None, None, None, None, None
    disp_wy = disp[mask]; rain_wy = rain[mask]; dates_wy = dates[mask]

    # Reset rainfall baseline
    if len(rain_wy) > 0:
        rain_wy = rain_wy - rain_wy[0]

    # Z-score normalize
    disp_norm = (disp_wy - np.mean(disp_wy)) / np.std(disp_wy) if np.std(disp_wy) > 0 else disp_wy
    rain_norm = (rain_wy - np.mean(rain_wy)) / np.std(rain_wy) if np.std(rain_wy) > 0 else rain_wy

    corr = correlate(disp_norm, rain_norm, mode="full")
    lags = correlation_lags(len(disp_norm), len(rain_norm), mode="full")

    # Fix timedelta -> float days
    step_days = np.median(np.diff(dates_wy).astype("timedelta64[D]").astype(float))
    lag_days = lags * step_days

    mask_lag = np.abs(lag_days) <= max_lag_days
    corr, lag_days = corr[mask_lag], lag_days[mask_lag]
    if len(corr) == 0:
        return None, None, None, dates_wy, None
    best_lag = lag_days[np.argmax(corr)]
    best_corr = np.max(corr)
    return lag_days, corr, best_lag, best_corr, dates_wy

# =========================
# Main group analysis
# =========================
def analyze_group(group_name, subdf):
    # Load all landslide TS and rainfall
    series_disp, series_rain = [], []
    for _,row in subdf.iterrows():
        try:
            dates, ts, rain = load_timeseries(row["file"])
            disp_norm = normalize_minmax(ts)
            disp_s = pd.Series(disp_norm, index=pd.to_datetime(dates))
            series_disp.append(disp_s)

            # daily rainfall
            rain_s = pd.Series(rain.values, index=pd.to_datetime(rain.index))
            series_rain.append(rain_s)
        except Exception:
            continue
    if len(series_disp)==0:
        return None

    disp_df = pd.concat(series_disp, axis=1).interpolate().sort_index()
    rain_df = pd.concat(series_rain, axis=1).interpolate().sort_index()
    disp_med = disp_df.median(axis=1)

    # Cumulative rainfall across all slides
    rain_cum_df = rain_df.cumsum()
    rain_med = []
    for d in disp_med.index:
        vals = rain_cum_df.loc[rain_cum_df.index <= d].median(axis=1)
        rain_med.append(vals.iloc[-1] if len(vals) else 0.0)
    rain_med = pd.Series(rain_med, index=disp_med.index)

    # Normalize cumulative rainfall (0–1), same data used for CC and plots
    rain_med = normalize_minmax(rain_med.values)
    rain_med = pd.Series(rain_med, index=disp_med.index)

    # Find periods
    period1,period2 = find_custom_splits(disp_med.index, rain_med.values)

    results = {"group": group_name, "n_series": len(series_disp)}

    # Plot layout: 2 rows × 2 cols (P1 left, P2 right)
    fig, axes = plt.subplots(2,2,figsize=(14,8), sharex=False)
    (ax_cc1, ax_cc2), (ax_ts1, ax_ts2) = axes

    for (period,label,ax_cc,ax_ts) in [
        (period1,"Period1",ax_cc1,ax_ts1),
        (period2,"Period2",ax_cc2,ax_ts2)]:

        lags,corr,best_lag,best_corr,dates_wy = cross_correlation(
            disp_med.index, disp_med.values, rain_med.values,
            period[0],period[1],max_lag_days=MAX_LAG_DAYS
        )

        # --- CC curve
        if lags is not None:
            ax_cc.plot(lags,corr,label="CC")
            ax_cc.axvline(best_lag,color="red",ls="--",label=f"Lag={best_lag:.1f}d")
            ax_cc.set_title(f"{label} CC"); ax_cc.legend()
            results[f"lag_{label}"]=best_lag
            results[f"corr_{label}"]=best_corr
        else:
            ax_cc.set_title(f"{label} CC: insufficient data")
            results[f"lag_{label}"]=np.nan
            results[f"corr_{label}"]=np.nan

        # --- Time series
        mask = (disp_med.index>=period[0])&(disp_med.index<=period[1])
        if mask.sum()>0:
            for col in disp_df.columns:
                ax_ts.scatter(disp_df.index, disp_df[col], s=2, color="gray", alpha=0.2)
            ax_ts.plot(disp_med.index,disp_med.values,"b-",lw=2,label="Median disp")
            q05=disp_df.quantile(0.05,axis=1); q95=disp_df.quantile(0.95,axis=1)
            ax_ts.fill_between(disp_med.index,q05,q95,alpha=0.3,color="blue",label="5–95%")
            ax_ts.plot(rain_med.index,rain_med.values,"g-",alpha=0.6,label="Cumulative rain (norm.)")
            ax_ts.set_xlim(period[0], period[1])  # match correlation window
            ax_ts.set_title(f"{label} time series"); ax_ts.legend()

    fig.suptitle(f"Group {group_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,f"group_{group_name}_correlation.png"),dpi=300)
    plt.close(fig)

    return results

# =========================
# Driver
# =========================
def main():
    df = pd.read_csv(CSV_PATH)
    df["vel_dry1"]=np.abs(df["meta__ts_dry1_vel_myr"]*100)
    df["vel_dry2"]=np.abs(df["meta__ts_dry2_vel_myr"]*100)
    df = df[(df["vel_dry1"]>=VEL_MIN_THRESHOLD)|(df["vel_dry2"]>=VEL_MIN_THRESHOLD)].copy()
    df["vel_ratio"]=df["vel_dry2"]/df["vel_dry1"]
    df["group"]=df["vel_ratio"].apply(categorize_ratio_group)

    # Map IDs to HDF5
    h5_files=glob.glob(os.path.join(H5_DIR,"*.h5"))
    h5_records=[]
    for fp in h5_files:
        try:
            with h5py.File(fp,"r") as hf:
                sid=hf["meta"].attrs.get("ID")
                h5_records.append({"ls_id":str(sid),"file":fp})
        except: continue
    df_h5=pd.DataFrame(h5_records)
    id_candidates=[c for c in df.columns if re.search(r"(ls_id|meta__ls_id|ID)$",c)]
    csv_id_col=id_candidates[0]
    df[csv_id_col]=df[csv_id_col].astype(str)
    merged=df.merge(df_h5,left_on=csv_id_col,right_on="ls_id",how="left")
    merged=merged[merged["file"].notna()]

    groups=["Much Faster","Faster","Similar","Slower","Much Slower",
            "Active (exclude Much Slower)"]
    results=[]
    for g in groups:
        if g=="Active (exclude Much Slower)":
            sub=merged[merged["group"]!="Much Slower"]
        else:
            sub=merged[merged["group"]==g]
        if len(sub)==0: continue
        res=analyze_group(g,sub)
        if res: results.append(res)

    csv_path = os.path.join(OUT_DIR,"group_lag_summary.csv")
    pd.DataFrame(results).to_csv(csv_path,index=False)
    print(f"Done. Results saved to {csv_path}")

if __name__=="__main__":
    main()
