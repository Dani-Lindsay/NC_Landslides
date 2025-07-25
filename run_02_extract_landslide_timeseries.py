#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:39:56 2025

@author: daniellelindsay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch‐process landslide ROIs, velocity statistics, and time‐series extraction
across all y*_box directories.
"""

import os
import glob
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt
import pyproj
from datetime import datetime
from scipy.stats import linregress
from NC_Landslides_paths import *

# -------------------------------------------------------------------------
# Constants & Defaults
# -------------------------------------------------------------------------
DEFAULT_RADIUS_M         = 700
BUFFER_M                 = 4000
#BUFFER_FACTOR            = 1.5
ROI_FACTOR               = 2.0
BACKGROUND_STD_THRESHOLD = 0.02
MIN_Q95_PIXELS           = 10           # For large landslides allow us get time series of fastest pixels
MIN_Q75_PIXELS           = 10           # Sets the miniumum resolveable size of landslide
DISTRIBUTION_THRESHOLD   = 100          # How scattered the pixels can be
poly_order               = 4            # Polynomial order for calculating RMSE 

dataset_label = "Timeseries_1"

eq1 = 2021.9685
eq2 = 2022.9685

eq_21_lon = -124.298
eq_21_lat = 40.390
eq_21_m = 6.2 
eq_22_lat = 40.525
eq_22_lon = -124.423
eq_22_m = 6.4

vel_t1 = 2022.1667
vel_t2 = 2022.9167
vel_t3 = 2023.1667
vel_t4 = 2023.9167

SKIP_IDS = {} #{"wc486", "wc107", "wc340", "wc341"}
STATS = {k:0 for k in [
    'manual_skip','no_valid_data','high_background_std',
    'few_q75_pixels','high_nn_distance','ts_q75', 'ts_q95'
]}



GEOD = pyproj.Geod(ellps='WGS84')

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def calc_azi(lon1, lat1, lon2, lat2):
    """Broadcasting azimuth/distance."""
    if np.isscalar(lon1):
        lon1 = np.full_like(lon2, lon1); lat1 = np.full_like(lat2, lat1)
    return GEOD.inv(lon1, lat1, lon2, lat2)

def find_nearest_idx(arr, val):
    return int(np.nanargmin(np.abs(np.asarray(arr)-val)))

def to_year_fraction(dt_obj):
    def since_epoch(d): return time.mktime(d.timetuple())
    start = datetime(dt_obj.year,1,1)
    end   = datetime(dt_obj.year+1,1,1)
    return dt_obj.year + (since_epoch(dt_obj)-since_epoch(start))/(since_epoch(end)-since_epoch(start))

def meters_to_deg(m, lat):
    return m / (111000 * np.cos(np.deg2rad(lat)))

def compute_roi_window(lon, lat, lon_1d, lat_1d, ncols, nrows, max_diam):
    """Return radius_m, buffer_m, x0,x1,y0,y1 raw window."""
    radius   = max(DEFAULT_RADIUS_M, max_diam/2.0)
    buffer_m = BUFFER_M
    half_ext = buffer_m * ROI_FACTOR
    dlon = meters_to_deg(half_ext, lat)
    dlat = meters_to_deg(half_ext, lat)
    x0 = max(find_nearest_idx(lon_1d, lon-dlon), 0)
    x1 = min(find_nearest_idx(lon_1d, lon+dlon)+1, ncols)
    y0 = max(find_nearest_idx(lat_1d, lat-dlat), 0)
    y1 = min(find_nearest_idx(lat_1d, lat+dlat)+1, nrows)
    return radius, buffer_m, x0, x1, y0, y1

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
def plot_median_with_iqr(ts_df, title, save_path):
    fig, ax = plt.subplots(figsize=(8,5))
    med, q1, q3 = ts_df['median_ls'], ts_df['q1_ls'], ts_df['q3_ls']
    ax.errorbar(ts_df.index, med, yerr=[med-q1,q3-med], fmt='o',
                ecolor='gray', capsize=2, label='Original')
    clean, el, eh = ts_df['clean_ts'], ts_df['err_low'], ts_df['err_high']
    ax.errorbar(ts_df.index, clean, yerr=[el,eh], fmt='s',
                ecolor='black', capsize=2, label='Cleaned')
    ax.set(title=title, xlabel="Decimal Year", ylabel="Velocity (m/yr)")
    ax.grid('--', alpha=0.5); ax.legend(); fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    

def plot_roi_map(
    box_id, ls_id, ls_row, roi_df, inside, inside_ab, ls_q75, ls_q95, ls_sign,
    ts_df,
    radius, buffer, poly_gmt, fig_dir, cluster_label, coeffs_clean
):
    fig_region = "%s/%s/%s/%s" % (np.nanmin(roi_df['Lon']), np.nanmax(roi_df['Lon']),
                                  np.nanmin(roi_df['Lat']), np.nanmax(roi_df['Lat']))

    bg_median = ls_row["background_median"]
    vmin = np.nanmin(roi_df['Vel'])*100
    vmax = np.nanmax(roi_df['Vel'])*100
    count_max = 250

    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",
                 FONT=9, FONT_TITLE=10, MAP_TITLE_OFFSET=0)
    pygmt.makecpt(series=[bg_median-0.05, bg_median+0.05], cmap="vik")

    fig.basemap(region=fig_region, projection="M5.9c",
                frame=["tSWr", 'xa0.04', 'ya0.02'])
    fig.plot(x=roi_df['Lon'], y=roi_df['Lat'], fill=roi_df['Vel'],
             style="c0.09c", cmap=True)
    fig.plot(x=ls_q75['Lon'], y=ls_q75['Lat'], fill="gold", style="c0.09c")
    if not ls_q95.empty:
        fig.plot(x=ls_q95['Lon'], y=ls_q95['Lat'],
                 fill="purple2", style="c0.09c")
    fig.plot(data=poly_gmt, pen="1.5p,black")
    fig.plot(x=ls_row['Lon'], y=ls_row['Lat'], size=[
             radius*2/1000], style="E-", pen="1.5p,purple2")
    fig.plot(x=ls_row['Lon'], y=ls_row['Lat'], size=[
             buffer*2/1000], style="E-", pen="1.5p,forestgreen")

    fig.text(text="%s" % ls_row['ID'], position="TC",
             offset="0.0c/-0.2c", font="14p,Helvetica-Bold")

    fig.basemap(region=fig_region, projection="M5.9c", frame=["tSWr", 'xa0.04', 'ya0.02'],
                map_scale="jTR+w1k+o0.4/0.4c",)

    fig.colorbar(cmap=True, frame=["x", "y+lm/yr"],
                 position="JBC+o0c/0.75c+w4c/0.4c+h")

    fig.shift_origin(xshift="7.5c")
    fig.histogram(data=inside["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                  frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="gold", transparency=50, label='inside radius')
    fig.histogram(data=inside_ab["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                  frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="orange", transparency=50, label='exluding location med+/-*2std')
    fig.histogram(data=ls_q75["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                  frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="red", transparency=50, label='q75')
    if not ls_q95.empty:
        fig.histogram(data=ls_q95["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                      frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="blue", transparency=50, label='q95')
    fig.legend(position="JBC+o0c/0.75c")

    fig.shift_origin(yshift="4.5c")
    fig.basemap(region=[0, 10, 0, 10], projection="X6c/3c", frame=["tblr"])
    # Left text
    if ls_sign < 0:
        fig.text(text="Negative", position="TL",
                 offset="0.2c/-0.2c", font="Helvetica")
    elif ls_sign > 0:
        fig.text(text="Postive", position="TL",
                 offset="0.2c/-0.2c", font="Helvetica")
        
    fig.text(text="Bg Std. %s cm" % (np.round(
        ls_row['background_std']*100, 1)), position="TL", offset="0.2c/-0.7c", font="Helvetica")
    
    fig.text(text="Dist. to ref %s km" % (np.round(
        ls_row['ref_dist']/1000, 1)), position="TL", offset="0.2c/-1.2c", font="Helvetica")


    fig.text(text="Pixels Q75: %s, Q95: %s" % (len(ls_q75), len(ls_q95)),
             position="TL", offset="0.2c/-1.7c", font="Helvetica")
    
    fig.text(text="NN Dist. %s m mean" % (np.round(
        ls_row['mean_nn'], 1)), position="TL", offset="0.2c/-2.2c", font="Helvetica")
    
    # Right text 
    fig.text(text="Radius: %s km" % (np.round(radius/1000, 1)),
             position="TL", offset="3c/-0.2c", font="Helvetica")
    
    fig.text(text="TS source: %s " % (cluster_label),
             position="TL", offset="3c/-0.7c", font="Helvetica")

    fig.shift_origin(yshift="-4.5c", xshift="7c")
    region_ts = [np.nanmin(ts_df["dates"])-0.2, np.nanmax(ts_df["dates"])+0.2, np.nanmin(ts_df['clean_ts'])-0.02, np.nanmax(ts_df['clean_ts'])+0.02]
    fig.basemap(region=region_ts, projection="X12c/7.5c", frame=["lStE", "xaf+lYears", "yaf+lLOS Displacement (cm)"])

    eq1 = 2021.9685
    eq2 = 2022.9685
    
    fig.plot(x=[2021.9167, 2021.9167,  2022.1667,  2022.1667, 2021.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2022.9167, 2022.9167,  2023.1667,  2023.1667, 2022.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)
    fig.plot(x=[2023.9167, 2023.9167,  2024.1667,  2024.1667, 2023.9167], y=[region_ts[2], region_ts[3], region_ts[3], region_ts[2], region_ts[2]],  fill="lightblue", transparency=50)    
    fig.plot(x=[eq1, eq1], y=[region_ts[2], region_ts[3]],  pen="1p,black,--",) 
    fig.plot(x=[eq2, eq2], y=[region_ts[2], region_ts[3]],  pen="1p,black,--",)
    
    for _, row in ts_df.iterrows():
        fig.plot(x=[row["dates"], row["dates"]], y=[ row['q1_ls'],  row['q3_ls']],
            pen="1p,darkorange", transparency=50)
        fig.plot(x=[row["dates"], row["dates"]], y=[row['clean_ts'] - row['err_low'], row['clean_ts'] + row['err_high']],
            pen="1p,dodgerblue4", transparency=50)

    # 1) make a nice dense set of x’s between min and max date
    x_fit = np.linspace(ts_df["dates"].min(), ts_df["dates"].max(), 200)
    
    # 2) evaluate the polynomial at those x’s
    #    coeffs_clean is like [a, b, c] from np.polyfit(...)
    y_fit = np.polyval(coeffs_clean, x_fit)

    # 3) plot the line
    fig.plot(x=x_fit, y=y_fit, pen="2p,dodgerblue4", label="2nd order fit", transparency=50)
    
    fig.plot(x=ts_df["dates"], y=ts_df["median_ls"], style="c0.15c",
             pen="1p,darkorange", fill="darkorange", label="Original")
    
    fig.plot(x=ts_df["dates"], y=ts_df["clean_ts"], style="c0.15c",
             pen="1p,dodgerblue4", fill="dodgerblue4", label="Cleaned")
    fig.legend()
    
    fname = f"{fig_dir}/{ls_row['ID']}_{box_id}_radius{np.round(radius,1)}_buffer{buffer}_{cluster_label}_{dataset_label}.png"
    fig.savefig(fname, transparent=False, crop=True,
                anti_alias=True, show=False)
    fig.show()
    print(f"   • saved TS → {fname}")


# -------------------------------------------------------------------------
# Time‐series Extraction + Plotting
# -------------------------------------------------------------------------
def extract_and_plot_timeseries(
    box_id, ls_id, ls_row, ls_q75,
    roi_lons, roi_lats,
    buffer_m, radius_m,
    ts_file, fig_dir,
    y0, y1, x0, x1,
    clean_df, err_low_df, err_high_df
):
    # 1) read dates + cube
    with h5py.File(ts_file,'r') as hf:
        raw_dates = hf['date'][:]
        dates = [ to_year_fraction(datetime.strptime(d.decode(),'%Y%m%d'))
                  for d in raw_dates ]
        ts_cube = hf['timeseries'][:, y0:y1, x0:x1]

    # 2) per-pixel TS
    pix_ts = {}
    for i, pt in ls_q75.iterrows():
        iy, ix = int(pt['y_loc']), int(pt['x_loc'])
        pix_ts[f"pix_{i}"] = ts_cube[:, iy, ix]
    ts_df = pd.DataFrame(pix_ts, index=dates)
    ts_df["dates"] = dates

    # 3) slide median/IQR
    ts_df['q1_ls']     = ts_df.quantile(0.25,axis=1)
    ts_df['q3_ls']     = ts_df.quantile(0.75,axis=1)
    ts_df['median_ls'] = ts_df.median(axis=1)

    # 4) background TS
    flat_lon = roi_lons.ravel(); flat_lat = roi_lats.ravel()
    _,_,d = calc_azi(ls_row['Lon'], ls_row['Lat'], flat_lon, flat_lat)
    mask_bg = d.reshape(roi_lats.shape) > buffer_m
    bg_ts = ts_cube[:, mask_bg]
    bg_cols = [f"bg_{i}" for i in np.where(mask_bg.ravel())[0]]
    ts_bg = pd.DataFrame(bg_ts, index=dates, columns=bg_cols)

    ts_df['q1_bg']     = ts_bg.quantile(0.25,axis=1)
    ts_df['q3_bg']     = ts_bg.quantile(0.75,axis=1)
    ts_df['median_bg'] = ts_bg.median(axis=1)

    # 5) cleaned TS + errors
    ts_df['clean_ts'] = ts_df['median_ls'] - ts_df['median_bg']
    e_ls_low  = ts_df['median_ls'] - ts_df['q1_ls']
    e_ls_high = ts_df['q3_ls']    - ts_df['median_ls']
    e_bg_low  = ts_df['median_bg'] - ts_df['q1_bg']
    e_bg_high = ts_df['q3_bg']    - ts_df['median_bg']
    ts_df['err_low']  = np.sqrt(e_ls_low**2  + e_bg_low**2)
    ts_df['err_high'] = np.sqrt(e_ls_high**2 + e_bg_high**2)

    # 6) store
    clean_df[ls_id]    = ts_df['clean_ts']
    err_low_df[ls_id]  = ts_df['err_low']
    err_high_df[ls_id] = ts_df['err_high']

    # 7) plot + save
    #out_png = os.path.join(fig_dir, f"{box_id}_{ls_id}_ts.png")
    #plot_median_with_iqr(ts_df, title=f"{box_id} {ls_id}", save_path=out_png)
    #print(f"   • saved TS → {out_png}")
    
    return ts_df, clean_df, err_low_df, err_high_df


def extract_geometry_means(geo_file, y0, y1, x0, x1, pixels_df):
    """
    Read all available grids in geo_file for the window [y0:y1,x0:x1],
    sample them at the (y_loc, x_loc) in pixels_df, and return a dict of their means.
    """
    want = [
        'height','slope','aspect',
        'incidenceAngle','azimuthAngle',
        'latitude','longitude',
        'slantRangeDistance','waterMask'
    ]
    means = {}
    with h5py.File(geo_file, 'r') as hf:
        # keep only those datasets actually present
        have = [v for v in want if v in hf]
        # load the window into memory once per dataset
        windows = {v: hf[v][y0:y1, x0:x1] for v in have}

        # for each field, sample at each pixel and compute its mean
        for v in have:
            vals = []
            arr = windows[v]
            for _, row in pixels_df.iterrows():
                iy, ix = int(row['y_loc']), int(row['x_loc'])
                vals.append(arr[iy, ix])
            means[v] = np.nanmean(vals)

    return means

def extract_mean_coherence(coh_file, pixels_df):
    """
    Read the full /coherence array from coh_file and return the mean
    coherence over the (y_loc, x_loc) pixels in pixels_df.
    
    Parameters
    ----------
    coh_file : str
        Path to the HDF5 holding a top‑level '/coherence' dataset.
    pixels_df : pandas.DataFrame
        Must contain integer columns 'y_loc' and 'x_loc'.
    
    Returns
    -------
    float
        The mean coherence (ignoring NaNs) over those pixels.
    """
    with h5py.File(coh_file, "r") as hf:
        coh = hf["coherence"][:]  # shape e.g. (1707,1644)
    # gather values at each requested pixel
    vals = []
    for _, row in pixels_df.iterrows():
        y, x = int(row["y_loc"]), int(row["x_loc"])
        vals.append(coh[y, x])
    return float(np.nanmean(vals))

def poly_rmse(dates, values, deg=4):
    """
    Fit a degree‐`deg` polynomial to (dates, values) and return
    both the fit coefficients and the RMSE of the residuals.
    
    Parameters
    ----------
    dates : array‑like, shape (N,)
        Decimal‐year times.
    values : array‑like, shape (N,)
        LOS velocity (m/yr) or displacement (cm), etc.
    deg : int
        Polynomial degree (e.g. 1 for linear trend, 2 for quadratic).
    
    Returns
    -------
    coeffs : ndarray, shape (deg+1,)
        Polynomial coefficients (highest power first).
    rmse : float
        sqrt(mean((values − poly(dates))**2)), ignoring NaNs.
    """
    # fit
    mask = np.isfinite(dates) & np.isfinite(values)
    c = np.polyfit(dates[mask], values[mask], deg)
    # evaluate
    fit = np.polyval(c, dates)
    resid = values - fit
    rmse = np.sqrt(np.nanmean(resid[mask]**2))
    return c, rmse

def compute_velocity(dates, values, start, stop):
    """
    Fit a straight line to ‘values’ vs. ‘dates’ between start and stop,
    automatically ignoring any NaNs, and return (velocity, σ_velocity).

    Parameters
    ----------
    dates : array-like of float
        Decimal-year timestamps.
    values : array-like of float
        Displacements (same length as dates).
    start, stop : float
        Decimal-year window over which to fit.

    Returns
    -------
    slope : float
        Best-fit rate (units of values per year).
    stderr : float
        Standard error of the slope.
    """
    # convert to arrays
    t = np.asarray(dates, dtype=float)
    d = np.asarray(values, dtype=float)

    # mask to window and non-NaN
    m = (t >= start) & (t <= stop) & ~np.isnan(d)
    if m.sum() < 2:
        raise ValueError(f"Not enough valid points in [{start}, {stop}]")

    x = t[m]
    y = d[m]

    res = linregress(x, y)
    return res.slope, res.stderr

# -------------------------------------------------------------------------
# Per‐landslide processing
# -------------------------------------------------------------------------
def process_landslide(ls_row, vel_lons, vel_lats, lon_1d, lat_1d,
                      ncols, nrows, ref_lon, ref_lat,
                      vel_file, ts_file, fig_dir,
                      clean_df, err_low_df, err_high_df,
                      box_id):

    ls_id, ls_lon, ls_lat = ls_row['ID'], ls_row['Lon'], ls_row['Lat']
    print(f"\n→ {ls_id}")
    if ls_id in SKIP_IDS:
        STATS['manual_skip'] += 1; print("   • skipped"); return

    # reference distance
    _,_,ref_d = calc_azi(ref_lon, ref_lat, ls_lon, ls_lat)
    ls_row['ref_dist'] = ref_d
    
    # eq1 distance
    _,_,eq1_dist = calc_azi(eq_21_lon, eq_21_lat, ls_lon, ls_lat)
    ls_row['eq_21_dist'] = eq1_dist
    
    # eq2 distance
    _,_,eq2_dist = calc_azi(eq_22_lon, eq_22_lat, ls_lon, ls_lat)
    ls_row['eq_22_dist'] = eq2_dist

    # raw ROI window
    r, b, x0, x1, y0, y1 = compute_roi_window(
        ls_lon, ls_lat, lon_1d, lat_1d, ncols, nrows,
        ls_row['max_diameter_m']
    )
    xs, xe = sorted((x0,x1)); ys, ye = sorted((y0,y1))
    print(f"   • r={r:.0f} buf={b:.0f} x[{xs}:{xe}] y[{ys}:{ye}]")

    # read velocity block
    roi_lons = vel_lons[ys:ye, xs:xe]
    roi_lats = vel_lats[ys:ye, xs:xe]
    with h5py.File(vel_file,'r') as hf:
        vel = hf['velocity'][ys:ye, xs:xe]
    vel[vel==0] = np.nan

    # flatten + vectorized distance + loc
    ny, nx = vel.shape
    ys_loc, xs_loc = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    flat_lon = roi_lons.ravel(); flat_lat = roi_lats.ravel()
    _,_,d = calc_azi(ls_lon, ls_lat, flat_lon, flat_lat)

    df = pd.DataFrame({
        'y_loc':    ys_loc.ravel(),
        'x_loc':    xs_loc.ravel(),
        'Lon':      flat_lon,
        'Lat':      flat_lat,
        'Vel':      vel.ravel(),
        'distance': d
    })

    # mask
    m = (df['distance']<r)&df['Vel'].notna()
    if not m.any():
        STATS['no_valid_data']+=1; print("   • no data"); return

    # background stats
    bg = df[df['distance']>b]
    med, std = np.nanmedian(bg['Vel']), np.nanstd(bg['Vel'])
    ls_row['background_median']=med; ls_row['background_std']=std
    if std>BACKGROUND_STD_THRESHOLD:
        STATS['high_background_std']+=1; print("   • high bg std"); return
        
    # Q75 cluster
    inside = df[df['distance']<r]
    
    # Keep only pixels above background std * 2
    inside_ab = inside[
        (inside['Vel'] > med + 2 * std) |
        (inside['Vel'] < med - 2 * std)
    ]
    
    # 1) compute deviant pixels at 75% & 95%
    ls_sign = np.sign(np.nanmedian(inside_ab['Vel'] - med))
    dev = (inside_ab['Vel'] - med) * ls_sign

    q75 = np.nanquantile(dev, 0.75)
    ls_q75 = inside_ab[dev > q75]

    q95 = np.nanquantile(dev, 0.95)
    ls_q95 = inside_ab[dev > q95]

    # … compute dev & both clusters …
    if len(ls_q95) >= 10:
        ls_cluster   = ls_q95
        cluster_label = "q95"
        print(f"   • using Q95 cluster ({len(ls_q95)} pixels)")
    else:
        if len(ls_q75) < MIN_Q75_PIXELS:
            STATS['few_q75_pixels'] += 1
            print("   • too few Q75 pixels; skipping")
            return
        ls_cluster    = ls_q75
        cluster_label = "q75"     
        print(f"   • using Q75 cluster ({len(ls_q75)} pixels)")

    # 3) nearest‐neighbor distances on chosen cluster
    coords = ls_cluster[['Lon','Lat']].values
    nn = []
    for i, (li, la) in enumerate(coords):
        other = np.delete(coords, i, axis=0)
        _, _, ds = calc_azi(li, la, other[:, 0], other[:, 1])
        nn.append(ds.min())
    mean_nn = np.mean(nn)
    ls_row['mean_nn'] = mean_nn

    if mean_nn > DISTRIBUTION_THRESHOLD:
        STATS['high_nn_distance'] += 1
        print("   • sparse cluster; skipping")
        return

    ts_df, clean_df, err_low_df, err_hi_df = extract_and_plot_timeseries(
        box_id, ls_id, ls_row, ls_cluster, roi_lons, roi_lats,
        b, r, ts_file, fig_dir, ys, ye, xs, xe,
        clean_df, err_low_df, err_high_df
    )
    
    # 4.a) compute goodness‐of‐fit metrics
    #    here we fit a quadratic to the cleaned timeseries:
    coeffs_clean, rmse_clean = poly_rmse(ts_df["dates"].values,
                                         ts_df["clean_ts"].values,
                                         deg=poly_order)
    
    #    optionally also on the raw median LS:
    coeffs_orig, rmse_orig = poly_rmse(ts_df["dates"].values,
                                       ts_df["median_ls"].values,
                                       deg=poly_order)
    
    vel_dry1, err_dry1 = compute_velocity(ts_df["dates"].values, ts_df["median_ls"].values, start=vel_t1, stop=vel_t2)
    vel_dry2, err_dry2 = compute_velocity(ts_df["dates"].values, ts_df["median_ls"].values, start=vel_t3, stop=vel_t4)
    linear_vel, linear_err = compute_velocity(ts_df["dates"].values, ts_df["median_ls"].values, start=np.nanmin(ts_df["dates"].values), stop=np.nanmax(ts_df["dates"].values))
    
    # percentage of RMSE improved with cleaning    
    pct_rmse_red = 100 * (rmse_orig - rmse_clean) / rmse_orig

    # ─── grab geometry fields at those same pixels ────────────────────────────
    # grab per‑pixel means of all geometry fields
    geo_file = os.path.join(box, "geo", "geo_geometryRadar.h5")
    geo_means = extract_geometry_means(geo_file, ys, ye, xs, xe, ls_cluster)

    coh_file = os.path.join(box, "geo", "geo_avgSpatialCoh.h5")
    mean_coh = extract_mean_coherence(coh_file, ls_cluster)
    
    # right after you have ts_df, clean_df, err_low_df, err_high_df, and meta…
    dates_arr    = ts_df["dates"].values                 # shape (N,)
    clean_arr    = clean_df[ls_id].values                # shape (N,)
    err_low_arr  = err_low_df[ls_id].values              # shape (N,)
    err_high_arr = err_high_df[ls_id].values             # shape (N,)
    # … compute your diagnostics dict …
    if cluster_label == "q95":
        STATS['ts_q95'] += 1
    else:
        STATS['ts_q75'] += 1
    
    # Plot time series 
    # plot_roi_map(
    #     box_id, ls_id, ls_row, df, inside, inside_ab, ls_q75, ls_q95, ls_sign,
    #     ts_df,
    #     r, b, poly_gmt, fig_dir, cluster_label, coeffs_clean
    # )

    # --- write one HDF5 per‐slide ---
    # build output filename
    fname = f"{ls['ID']}_{box_id}_{cluster_label}_{dataset_label}.h5"
    out_h5 = os.path.join(data_dir, fname)

    # now write exactly one HDF5 per slide
    with h5py.File(out_h5, 'w') as hf:
        # 1) params group
        p = hf.create_group("params")
        for k,v in {
            'radius_m':          r,
            'buffer_m':          b,
            'bg_std_thresh_my-1':     BACKGROUND_STD_THRESHOLD,
            'min_num_q75':           MIN_Q75_PIXELS,
            'min_num_q95':           MIN_Q95_PIXELS,
            'nn_dist_thresh_m':       DISTRIBUTION_THRESHOLD,

        }.items():
            p.attrs[k] = v

        # 2) meta group
        m = hf.create_group("meta")
        # include *all* slide‐level info here
        slide_info = {
          'ID':                  ls['ID'],
          'Lat':                 ls['Lat'],
          'Lon':                 ls['Lon'],
          'ls_sign':             ls_sign,
          'ls_area_m2':          ls['area_m2'],
          'ls_perimeter_m':      ls['perimeter_m'],
          'ls_compactness':      ls['compactness'],
          'ls_max_diameter_m':   ls['max_diameter_m'],
          'ls_min_diameter_m':   ls['min_diameter_m'],
          'ls_axis_ratio':       ls['axis_ratio'],
          'ls_orientation_deg':  ls['orientation_deg'],
          'ls_mean_height':            geo_means.get('height', np.nan),
          'ls_mean_slope':             geo_means.get('slope',  np.nan),
          'ls_mean_aspect':            geo_means.get('aspect', np.nan),
          'ls_mean_incidenceAngle':    geo_means.get('incidenceAngle', np.nan),
          'ls_mean_avgSpatialCoh':    mean_coh,
          'sources':             ls['sources'],
          'dist_ref_m':            ls['ref_dist'],
          'dist_eq_21_m':          ls['eq_21_dist'],
          'dist_eq_22_m':          ls['eq_22_dist'],
          'ts_background_std_my-1':   ls['background_std'],
          'ts_mean_nn_dist_m':          mean_nn,
          'ts_num_q75':            len(ls_q75),
          'ts_num_q95':            len(ls_q95),
          'ts_data_label':    cluster_label,
          'ts_rmse_clean_m':    rmse_clean,   # if clean_ts is in m/yr, multiply
          'ts_rmse_orig_m':     rmse_orig,
          'ts_poly_deg':         poly_order,
          'ts_poly_coeffs_clean': coeffs_clean.tolist(),
          'ts_poly_coeffs_orig':  coeffs_orig.tolist(),
          'ts_pct_rmse_red':        pct_rmse_red, 
          'ts_linear_vel_myr':      linear_vel,
          'ts_linear_err_myr':      linear_err,
          'ts_dry1_vel_myr':        vel_dry1,
          'ts_dry1_err_myr':        err_dry1,
          'ts_dry2_vel_myr':        vel_dry2,
          'ts_dry2_err_myr':        err_dry2,
        }
        for k,v in slide_info.items():
            m.attrs[k] = v
            
        # 3) time‐series at root
        hf.create_dataset("dates",     data=dates_arr,    dtype='f8')
        hf.create_dataset("clean_ts",  data=clean_arr,    dtype='f8')
        hf.create_dataset("err_low",   data=err_low_arr,  dtype='f8')
        hf.create_dataset("err_high",  data=err_high_arr, dtype='f8')

    print(f"✔  wrote {out_h5}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__=="__main__":
    root     = ts_dir
    poly_csv = common_paths['ls_inventory']
    poly_gmt = common_paths['ls_gmt']['Xu_2021']
    fig_dir  = fig_dir
    data_dir = ts_out_dir
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ts_out_dir, exist_ok=True)

    # load landslide metrics
    slides = pd.read_csv(poly_csv, dtype={'ls_id':str})
    slides.rename(columns={'ls_id':'ID','center_lon':'Lon','center_lat':'Lat'}, inplace=True)

    boxes = sorted(glob.glob(os.path.join(root,"y*_box")))

    for box in boxes:
        box_id = os.path.basename(box)
        print(f"\n=== BOX {box_id} ===")

        geo_file = os.path.join(box,"geo","geo_geometryRadar.h5")
        vel_file = os.path.join(box,"geo","geo_velocity.h5")
        ts_list  = glob.glob(os.path.join(box,"geo","geo_timeseries*.h5"))
        if not ts_list:
            print("  • no TS file"); continue
        ts_file = ts_list[0]

        # read grid
        with h5py.File(geo_file,'r') as hf:
            vel_lons = hf["longitude"][:]
            vel_lats = hf["latitude"][:]
        nrows,ncols = vel_lats.shape
        lon_1d = np.nanmean(vel_lons, axis=0)
        lat_1d = np.nanmean(vel_lats, axis=1)

        # read dates & reference
        with h5py.File(ts_file,'r') as hf:
            raw_d = hf['date'][:]
            ref_lon = float(hf.attrs['REF_LON'])
            ref_lat = float(hf.attrs['REF_LAT'])
        dates_fixed = [to_year_fraction(datetime.strptime(d.decode(),'%Y%m%d')) for d in raw_d]

        # init TS DataFrames
        clean_df   = pd.DataFrame(index=dates_fixed)
        err_low_df = pd.DataFrame(index=dates_fixed)
        err_hi_df  = pd.DataFrame(index=dates_fixed)

        # filter slides in this box
        subset = slides[
            (slides['Lon'] >= lon_1d.min()) & (slides['Lon'] <= lon_1d.max()) &
            (slides['Lat'] >= lat_1d.min()) & (slides['Lat'] <= lat_1d.max())
        ].copy()
        print(f"  • {len(subset)} slides")

        meta_data = []
        
        for _, ls in subset.iterrows():
            process_landslide(
                ls, vel_lons, vel_lats, lon_1d, lat_1d,
                ncols, nrows, ref_lon, ref_lat,
                vel_file, ts_file, fig_dir,
                clean_df, err_low_df, err_hi_df,
                box_id
            )

    # Summary pie chart
    labels = ['Manual skip','No data','High bg std','Too few pxl','Sparse','TS q75', 'TS q95']
    sizes  = [STATS[k] for k in [
        'manual_skip','no_valid_data','high_background_std',
        'few_q75_pixels','high_nn_distance', 'ts_q75', 'ts_q95'
    ]]
    colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#8b8b8b', '#ffd92f']
    def ap(p): return f"{p:.1f}%" if p>=10 else ""
    fig,ax = plt.subplots(figsize=(7,4))
    wedges,_,_ = ax.pie(sizes, colors=colors, startangle=90,
                        autopct=ap, pctdistance=0.6,
                        wedgeprops=dict(edgecolor='white'))
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.05,0.5))
    ax.axis('equal')
    plt.title("Processing Outcomes", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"processing_outcomes.png"),dpi=300,bbox_inches='tight')
    plt.show()
    
    # Write out STATS to a text file
    stats_path = os.path.join(data_dir, "processing_stats.txt")
    with open(stats_path, 'w') as sf:
        for key, val in STATS.items():
            sf.write(f"{key}: {val}\n")
    print(f"Processing stats saved to {stats_path}")
