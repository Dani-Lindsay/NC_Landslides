import glob
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from NC_Landslides_paths import *

# — Path to your HDF5 time series files —
DEST_DIR = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_2"
ts_files = glob.glob(f"{DEST_DIR}/*.h5")

# — Flatten all series into a long DataFrame with date and normalized value —
records = []
for fpath in ts_files:
    with h5py.File(fpath, 'r') as h5:
        ts = h5['clean_ts'][:]
        dates = h5['dates'][:]
        sign = h5['meta'].attrs.get('ls_sign', 1.0)

    ts0 = ts - np.nanmean(ts)              # zero‐mean
    ts_signed = ts0 * sign                 # apply sign
    mn, mx = np.nanmin(ts_signed), np.nanmax(ts_signed)
    span = mx - mn
    ts_norm = (ts_signed - mn) / span if span else np.zeros_like(ts_signed)

    for d, v in zip(dates, ts_norm):
        records.append({'date': d, 'value': v})

df = pd.DataFrame(records)

# — Compute summary stats per date —
summary = (
    df.groupby('date')['value']
      .agg(median=lambda x: np.nanmedian(x),
           p5=lambda x: np.nanpercentile(x, 5),
           p95=lambda x: np.nanpercentile(x, 95))
      .reset_index()
      .sort_values('date')
)

# — Apply rolling smoothing (window in number of samples) —
window = 5
summary['median_smooth'] = summary['median'].rolling(window, center=True, min_periods=1).mean()
summary['p5_smooth']    = summary['p5'].rolling(window, center=True, min_periods=1).mean()
summary['p95_smooth']   = summary['p95'].rolling(window, center=True, min_periods=1).mean()

# — Plot —
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 8))

# 1) all points as light grey dots
ax.scatter(df['date'], df['value'], color='grey', alpha=0.3, s=10)

# 2) smoothed median line
ax.plot(summary['date'], summary['median_smooth'],
        color='tab:blue', linewidth=2, label='Median (smoothed)')

# 3) shaded smoothed 5–95% band
ax.fill_between(summary['date'],
                summary['p5_smooth'],
                summary['p95_smooth'],
                color='tab:blue', alpha=0.2,
                label='5–95th percentile (smoothed)')

# — Labels and title —
ax.set_xlabel('Year')
ax.set_ylabel('Normalized Displacement (0→1)')
ax.set_title('Landslide Time Series: Smoothed Median & Bounds')

# — Despine only top/right to keep full bottom spine —
sns.despine(ax=ax, top=True, right=True)

# If you wanted to trim but extend the bottom spine manually:
# sns.despine(ax=ax, trim=True)
# ax.spines['bottom'].set_bounds(df['date'].min(), df['date'].max())

ax.legend()
plt.tight_layout()

# — Save and show —
plt.savefig(f'{fig_dir}/timeseries_smoothed_fullspine.png', dpi=300, bbox_inches='tight')
plt.show()
