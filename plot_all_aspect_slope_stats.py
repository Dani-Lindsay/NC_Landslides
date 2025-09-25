#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple aspect rose & slope histogram, plus summary stats.
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt

# User parameters
root_dir    = "/Volumes/Seagate/NC_Landslides/Timeseries_2"
slope_min   = 4         # only include slopes >= 4°
aspect_bins = 36        # number of rose bins
slope_bins  = 30        # number of histogram bins
save_path   = "/Volumes/Seagate/NC_Landslides/Figures/Aug6/aspect_slope_simple.png"

# 1. Gather & filter data
geo_files = glob.glob(f"{root_dir}/**/geo/geo_geometryRadar.h5", recursive=True)
if not geo_files:
    geo_files = [f"{root_dir}/geo_geometryRadar.h5"]

all_aspects, all_slopes = [], []
for fn in geo_files:
    with h5py.File(fn, 'r') as f:
        a = f['aspect'][:].ravel()
        s = f['slope'][:].ravel()
    mask = s >= slope_min
    all_aspects.append(a[mask])
    all_slopes.append(s[mask])

aspects = np.concatenate(all_aspects)
slopes  = np.concatenate(all_slopes)
aspects_rad = np.deg2rad(aspects)

# 2. Plot
fig = plt.figure(figsize=(6, 10), constrained_layout=False)
# tighten margins so the histogram matches the rose diameter
fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05, hspace=0.3)

# 2a. Aspect rose
ax1 = fig.add_subplot(2, 1, 1, projection='polar')
theta_bins = np.linspace(0, 2*np.pi, aspect_bins+1)
ax1.hist(aspects_rad, bins=theta_bins, color='C0', edgecolor='white')
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_title(f'Aspect Rose (Slope ≥ {slope_min}°)', pad=20)
ax1.set_rticks([])

# 2b. Slope histogram
ax2 = fig.add_subplot(2, 1, 2)
ax2.hist(slopes, bins=slope_bins, color='C0', edgecolor='white')
ax2.set_xlabel('Slope (°)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Slope Histogram (Slope ≥ {slope_min}°)')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# remove the outside box on the slope axes
for spine in ax2.spines.values():
    spine.set_visible(False)

# 3. Save figure
fig.savefig(save_path, dpi=300)
print(f"Figure saved to {save_path}")

# 4. Summary statistics (as before)
mean_s   = np.nanmean(slopes)
med_s    = np.nanmedian(slopes)
min_s    = np.nanmin(slopes)
max_s    = np.nanmax(slopes)
print("\n=== SUMMARY STATISTICS ===")
print(f"Mean slope:   {mean_s:.2f}°")
print(f"Median slope: {med_s:.2f}°")
print(f"Slope range:  {min_s:.1f}°–{max_s:.1f}°")

pct_b20 = 100*np.sum(slopes<20)/len(slopes)
pct_E   = 100*np.sum((slopes>20)&(slopes<=25))/len(slopes)
pct_F   = 100*np.sum((slopes>=26)&(slopes<=35))/len(slopes)
pct_G   = 100*np.sum(slopes>35)/len(slopes)
print(f"\n{pct_b20:.1f}% of slopes <20°, then {pct_E:.1f}% moderately steep (20–25°), "
      f"{pct_F:.1f}% steep (26–35°), {pct_G:.1f}% very steep (>35°).")

ew = ((aspects>=45)&(aspects<135))|((aspects>=225)&(aspects<315))
ns = ((aspects>=337.5)|(aspects<22.5))|((aspects>=157.5)&(aspects<202.5))
ewc = np.sum(ew); nsc = np.sum(ns)
rel_ew = 100*ewc/(ewc+nsc); rel_ns = 100*nsc/(ewc+nsc)
print(f"\nAmong cardinal slopes: EW {rel_ew:.1f}%, NS {rel_ns:.1f}% "
      f"({rel_ew-rel_ns:.1f}% more EW than NS)")

plt.show()
