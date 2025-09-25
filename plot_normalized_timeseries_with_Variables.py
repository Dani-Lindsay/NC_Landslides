import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NC_Landslides_paths import *
import os, re
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NC_Landslides_paths import *

# ── 1) scan all per‐slide HDF5s and build metadata table ────────────────
DEST_DIR = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"
h5_files = glob.glob(f"{DEST_DIR}/*.h5")

records = []
for fp in h5_files:
    with h5py.File(fp, "r") as hf:
        m = hf["meta"].attrs
        records.append({
            "file":               fp,
            "ls_id":              m["ID"],
            "ls_area_m2":         m["ls_area_m2"],
            "ts_linear_vel_myr":  m["ts_linear_vel_myr"],
            "ts_dry1_vel_myr":    m["ts_dry1_vel_myr"],
            "ts_dry2_vel_myr":    m["ts_dry2_vel_myr"],
            "ls_mean_slope":      m["ls_mean_slope"],
        })
df = pd.DataFrame(records)
df["vel_cm_yr"] = df["ts_linear_vel_myr"] * 100   # convenience

# ── 2) helper routines for loading & normalizing ────────────────────────
def load_timeseries(fpath):
    with h5py.File(fpath, "r") as hf:
        dates = hf["dates"][:]           # decimal‐year
        ts    = hf["clean_ts"][:]        # LOS in meters
        sign  = hf["meta"].attrs.get("ls_sign", 1.0)
    return dates, ts, sign

def normalize_ts(dates, ts, sign=1.0, method="minmax"):
    ts0 = (ts - np.nanmean(ts)) * sign
    if method=="minmax":
        mn, mx = np.nanmin(ts0), np.nanmax(ts0)
        normed = (ts0-mn)/(mx-mn) if mx>mn else np.zeros_like(ts0)
    elif method=="zscore":
        s = np.nanstd(ts0)
        normed = (ts0-np.nanmean(ts0))/s if s>0 else np.zeros_like(ts0)
    else:
        raise ValueError(method)
    return pd.Series(normed, index=dates)

def plot_normalized_group(df_sel, title, fname, method="minmax", window=5):
    """
    df_sel : DataFrame with at least a 'file' column
    title  : plot title
    fname  : outfile name 
    method : 'minmax' or 'zscore'
    window : smoothing window (in samples) for median+band
    """
    # load & normalize each TS
    series_list = []
    for _, row in df_sel.iterrows():
        dates, ts, sign = load_timeseries(row["file"])
        s = normalize_ts(dates, ts, sign, method=method)
        series_list.append(s.rename(row["ls_id"]))
    # combine into wide DataFrame
    df_ts = pd.concat(series_list, axis=1)
    
    # compute point‑cloud for scatter
    long = df_ts.stack().reset_index()
    long.columns = ["date","ls_id","value"]
    
    # summary stats per date
    summary = long.groupby("date")["value"].agg(
        median=lambda x: np.nanmedian(x),
        p5=lambda     x: np.nanpercentile(x,5),
        p95=lambda    x: np.nanpercentile(x,95),
    ).sort_index()
    # smooth
    summary["med_sm"] = summary["median"].rolling(window,center=True,min_periods=1).mean()
    summary["p5_sm"]  = summary["p5"   ].rolling(window,center=True,min_periods=1).mean()
    summary["p95_sm"] = summary["p95"  ].rolling(window,center=True,min_periods=1).mean()
    
    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(long["date"], long["value"],
               color="lightgray", s=8, alpha=0.4, label="_nolegend_")
    ax.plot(summary.index, summary["med_sm"],
            color="C0", lw=2, label="Median (smoothed)")
    ax.fill_between(summary.index,
                    summary["p5_sm"], summary["p95_sm"],
                    color="C0", alpha=0.3, label="5–95% band")
    
    # add count badge
    n_series = df_ts.shape[1]
    ax.text(
        0.02, 0.98,
        f"N = {n_series}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    
    ax.set_title(title, pad=12)
    ax.set_xlabel("Decimal Year")
    ax.set_ylabel("Normalized TS")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fname = os.path.join(fig_dir, f"norm_timeseries_{fname}.png")

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"→ saved figure to {fname}")

    plt.show()


# ── 4) pick your subsets ────────────────────────────────────────────────

area_thresh   = 1e6      # m²
vel_thresh    = 5.0      # cm/yr
slope_thresh  = 20.0     # degrees

top_n = 15
largest_slides   = df[df["ls_area_m2"]    > 3000000]
smallest_slides  = df[df["ls_area_m2"]    < 10000]
fastest_slides   = df[np.abs(df["vel_cm_yr"])     > 8]
slowest_slides   = df[np.abs(df["vel_cm_yr"])     < 2]
dry1_gt_dry2     = df[df.ts_dry1_vel_myr > df.ts_dry2_vel_myr]
dry2_gt_dry1     = df[df.ts_dry2_vel_myr > df.ts_dry1_vel_myr]
steepest_slides  = df[df["ls_mean_slope"]  > 30 ]
shallowest_slides=  df[df["ls_mean_slope"] < 12 ]

# ── define your thresholds ───────────────────────────────────────────────


# ── build boolean subsets ────────────────────────────────────────────────
by_area      = df[df["ls_area_m2"]    > area_thresh]
by_fast_vel  = df[df["vel_cm_yr"]     > vel_thresh]
dry1_gt_dry2 = df[df["ts_dry1_vel_myr"] > df["ts_dry2_vel_myr"]]
steep_slides = df[df["ls_mean_slope"]  > slope_thresh]

# ── 5) now plot each ────────────────────────────────────────────────────
plot_normalized_group(largest_slides,    "Largest Slides by Area > 3000000 m²", "largest_slides")
plot_normalized_group(smallest_slides,   "Smallest Slides by Area < 10000 m²", "smallest_slides")
plot_normalized_group(fastest_slides,    "Fastest Slides > 8 cm/yr", "fastest_slides")
plot_normalized_group(slowest_slides,    "Slowest Slides < 2cm/yr", "slowest_slides")
plot_normalized_group(dry1_gt_dry2,      "Slides: Dry1 > Dry2", "dry1_gt_dry2")
plot_normalized_group(dry2_gt_dry1,      "Slides: Dry2 > Dry1", "dry2_gt_dry1")
plot_normalized_group(steepest_slides,   "Steepest Slides > 30 degrees", "steepest_slides")
plot_normalized_group(shallowest_slides, "Shallowest Slides < 12 degrees", "shallowest_slides")
