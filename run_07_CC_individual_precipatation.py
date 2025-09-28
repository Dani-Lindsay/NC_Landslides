#!/usr/bin/env python3
"""
Cross-correlation between displacement and rainfall with custom seasonal splits.
- Splits: (1) start → last zero-rainfall of 2022 dry season
          (2) first dry period in 2022 → start of 2023 winter rains
- Saves per-landslide plots, lag_summary.csv, and summary scatter/histograms.
"""

import os, glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

# Earthquake dates (decimal → datetime)
eq1_dt = pd.to_datetime(2021.9685, format="%Y")
eq2_dt = pd.to_datetime(2022.9685, format="%Y")

# ----------------- CONFIG -----------------
# Directory where .h5 files live
H5_DIR = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"

# Where to save outputs
OUT_DIR = "/Volumes/Seagate/NC_Landslides/Figures/Correlation"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LAG_DAYS = 120   # restrict lags
DETREND = False      # leave off

# ----------------- HELPERS -----------------
def decimal_year_to_datetime(dyear):
    year = int(dyear)
    remainder = dyear - year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return start + (end - start) * remainder

def detrend(series):
    x = np.arange(len(series))
    mask = np.isfinite(series)
    if mask.sum() < 2:
        return series
    coeffs = np.polyfit(x[mask], series[mask], 1)
    return series - np.polyval(coeffs, x)

def load_timeseries_from_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        disp = f["clean_ts"][:]
        sign = f["meta"].attrs.get("ls_sign", 1.0)
        disp = disp * sign
        disp_dates = np.array([decimal_year_to_datetime(d) for d in f["dates"][:]])

        rain_dates = pd.to_datetime(f["rainfall/date"][:].astype(str)).tz_localize(None)
        rain_mm = f["rainfall/rain_mm"][:]
        rain_series = pd.Series(rain_mm, index=rain_dates).sort_index()
        cum_rain = np.array([rain_series.loc[:d].sum() for d in disp_dates])

    return disp_dates, disp, cum_rain

def find_custom_splits(dates, rain):
    """Find two custom seasonal periods based on rainfall with fallbacks."""
    rain = np.array(rain)

    # Period 1: start -> last zero rainfall in 2022
    mask_2022 = (pd.Series(dates).dt.year == 2022)
    idx_last_zero_2022 = np.where((rain == 0) & mask_2022)[0]
    if len(idx_last_zero_2022) > 0:
        end1 = dates[idx_last_zero_2022[-1]]
    else:
        # fallback: mid-2022
        end1 = pd.Timestamp("2022-07-01")
    start1 = dates[0]

    # Period 2: first dry period mid-2022 -> start of 2023 rains
    mask_mid2022 = (dates > pd.Timestamp("2022-06-01")) & (dates < pd.Timestamp("2022-12-01"))
    idx_zero = np.where((rain == 0) & mask_mid2022)[0]
    if len(idx_zero) > 1 and np.any(np.diff(idx_zero) == 1):
        start2 = dates[idx_zero[np.where(np.diff(idx_zero) == 1)[0][0]]]
    else:
        start2 = pd.Timestamp("2022-07-01")

    mask_autumn2023 = (dates > pd.Timestamp("2023-09-01"))
    idx_rain2023 = np.where((rain > 0) & mask_autumn2023)[0]
    if len(idx_rain2023) > 0:
        end2 = dates[idx_rain2023[0]]
    else:
        end2 = dates[-1]

    return (start1, end1), (start2, end2)


def cross_correlation(dates, disp, rain, start, end, max_lag_days=120):
    mask = (dates >= start) & (dates <= end)
    if mask.sum() < 5:
        return None, None, None, dates[mask], (disp[mask], rain[mask]), None

    disp_wy = disp[mask]
    rain_wy = rain[mask]
    dates_wy = dates[mask]

    # reset rainfall baseline
    if len(rain_wy) > 0 and np.isfinite(rain_wy[0]):
        rain_wy = rain_wy - rain_wy[0]

    # detect rainfall onset
    try:
        onset_idx = np.where(rain_wy > 0)[0][0]
        rain_onset = dates_wy[onset_idx]
    except IndexError:
        rain_onset = None

    # detrend if needed
    if DETREND:
        disp_wy = detrend(disp_wy)
        rain_wy = detrend(rain_wy)

    # normalize
    disp_wy = (disp_wy - np.mean(disp_wy)) / np.std(disp_wy)
    rain_wy = (rain_wy - np.mean(rain_wy)) / np.std(rain_wy)

    corr = correlate(disp_wy, rain_wy, mode="full")
    lags = correlation_lags(len(disp_wy), len(rain_wy), mode="full")
    lag_days = lags * (np.median(np.diff(dates_wy)).days)

    mask_lag = np.abs(lag_days) <= max_lag_days
    corr = corr[mask_lag]
    lag_days = lag_days[mask_lag]

    if len(corr) == 0:
        return None, None, None, dates_wy, (disp_wy, rain_wy), rain_onset

    best_lag = lag_days[np.argmax(corr)]
    return lag_days, corr, best_lag, dates_wy, (disp_wy, rain_wy), rain_onset

def analyze_and_plot(file_path):
    dates, disp, rain = load_timeseries_from_hdf5(file_path)
    period1, period2 = find_custom_splits(dates, rain)

    lags1, corr1, lag1, dates1, series1, onset1 = cross_correlation(dates, disp, rain, *period1)
    lags2, corr2, lag2, dates2, series2, onset2 = cross_correlation(dates, disp, rain, *period2)

    # -------- Plot --------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top row: correlation curves
    for ax, (lags, corr, lag, title) in zip(
        axes[0],
        [(lags1, corr1, lag1, "Period 1"), (lags2, corr2, lag2, "Period 2")],
    ):
        if lags is None:
            ax.set_title(f"{title}: Not enough data")
            continue
        ax.plot(lags, corr, label="Cross-correlation (normalized)")
        ax.axvline(lag, color="red", linestyle="--", label=f"Lag = {lag:.1f} d")
        ax.set_title(title)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Correlation")
        ax.legend()

    # Bottom row: time series
    for ax, (dates_wy, lag, series, onset, title) in zip(
        axes[1],
        [(dates1, lag1, series1, onset1, "Period 1"), (dates2, lag2, series2, onset2, "Period 2")],
    ):
        if dates_wy is None:
            ax.set_title(f"{title}: Not enough data")
            continue
        disp_wy, rain_wy = series
        ax.plot(dates_wy, disp_wy, "b-", label="Displacement (normalized)")
        ax.plot(dates_wy, rain_wy, "g-", alpha=0.6, label="Rainfall (normalized)")

        if lag is not None:
            shifted_dates = dates_wy + pd.to_timedelta(lag, unit="D")
            ax.plot(shifted_dates, rain_wy, "g--", alpha=0.7, label="Rainfall shifted")
            ax.text(0.02, 0.9, f"Lag = {lag:.1f} days", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        if onset is not None:
            ax.axvline(onset, color="green", linestyle=":", lw=1.5, label="Rain onset")

        ax.set_title(f"{title}: Time series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.legend()

    fig.suptitle(f"Cross-correlation and Time Series: {os.path.basename(file_path)}")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{os.path.basename(file_path)}_correlation.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return {"file": os.path.basename(file_path), "lag_Period1_days": lag1, "lag_Period2_days": lag2}

def process_multiple(files, out_csv="lag_summary.csv"):
    results = []
    for f in files:
        results.append(analyze_and_plot(f))
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, out_csv), index=False)
    return df

def summary_plots(csv_path):
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=["lag_Period1_days", "lag_Period2_days"], how="all")
    df_nonzero = df_clean[(df_clean["lag_Period1_days"] != 0) | (df_clean["lag_Period2_days"] != 0)]

    # Scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df_clean["lag_Period1_days"], df_clean["lag_Period2_days"], alpha=0.6)
    axes[0].plot([0, 120], [0, 120], "k--", alpha=0.5)
    axes[0].axhline(40, color="red", linestyle="--", alpha=0.5)
    axes[0].axvline(40, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlim(-5, 125); axes[0].set_ylim(-5, 125)
    axes[0].set_title("All landslides")
    axes[0].set_xlabel("Lag Period1 (days)"); axes[0].set_ylabel("Lag Period2 (days)")

    axes[1].scatter(df_nonzero["lag_Period1_days"], df_nonzero["lag_Period2_days"], alpha=0.6, color="tab:blue")
    axes[1].plot([0, 120], [0, 120], "k--", alpha=0.5)
    axes[1].axhline(40, color="red", linestyle="--", alpha=0.5)
    axes[1].axvline(40, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlim(-5, 125); axes[1].set_ylim(-5, 125)
    axes[1].set_title("Subset: at least one non-zero")
    axes[1].set_xlabel("Lag Period1 (days)"); axes[1].set_ylabel("Lag Period2 (days)")

    plt.suptitle("Cross-correlation lag times: Period1 vs Period2")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, "lag_scatter_summary.png"), dpi=300)
    plt.close(fig)

    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes[0,0].hist(df_clean["lag_Period1_days"].dropna(), bins=20, alpha=0.7, label="Period1")
    axes[0,0].axvline(40, color="red", linestyle="--"); axes[0,0].legend(); axes[0,0].set_title("All - Period1")
    axes[0,1].hist(df_clean["lag_Period2_days"].dropna(), bins=20, alpha=0.7, color="tab:orange", label="Period2")
    axes[0,1].axvline(40, color="red", linestyle="--"); axes[0,1].legend(); axes[0,1].set_title("All - Period2")
    axes[1,0].hist(df_nonzero["lag_Period1_days"].dropna(), bins=20, alpha=0.7, label="Period1")
    axes[1,0].axvline(40, color="red", linestyle="--"); axes[1,0].legend(); axes[1,0].set_title("Non-zero - Period1")
    axes[1,1].hist(df_nonzero["lag_Period2_days"].dropna(), bins=20, alpha=0.7, color="tab:orange", label="Period2")
    axes[1,1].axvline(40, color="red", linestyle="--"); axes[1,1].legend(); axes[1,1].set_title("Non-zero - Period2")

    plt.suptitle("Lag distributions: Period1 vs Period2 (All vs Non-zero subset)", y=0.95)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(os.path.join(OUT_DIR, "lag_hist_summary.png"), dpi=300)
    plt.close(fig)

# ----------------- MAIN -----------------
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))  # first 20
    summary = process_multiple(files, out_csv="lag_summary.csv")
    print(summary)
    summary_plots(os.path.join(OUT_DIR, "lag_summary.csv"))
    print(f"Saved results in {OUT_DIR}")
