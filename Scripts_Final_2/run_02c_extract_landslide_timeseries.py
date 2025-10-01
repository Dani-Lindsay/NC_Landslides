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

i need to check these ones
conflicts = {"ls_128","ls_129","ls_158","ls_184","ls_185","ls_186","ls_224","ls_226","ls_233","ls_234",
             "ls_237","ls_241","ls_247","ls_251","ls_253","ls_256","ls_263","ls_266","ls_267","ls_270",
             "ls_271","ls_272","ls_280","ls_298","ls_300","ls_301","ls_307","ls_312","ls_313","ls_315",
             "ls_316","ls_320","ls_322","ls_349","ls_365","ls_366","ls_368","ls_416","ls_437","ls_445",
             "ls_447","ls_448","ls_449","ls_452","ls_477"}
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
from pyproj import Geod, Transformer
from datetime import datetime
from scipy.stats import linregress
import math
from NC_Landslides_paths import *


# Change for each instance of Timeseries_? 
dataset_label = "Timeseries_1"
ts_dir = f"/Volumes/Seagate/NC_Landslides/{dataset_label}"

# -------------------------------------------------------------------------
# Constants & Defaults
# -------------------------------------------------------------------------
DEFAULT_RADIUS_M         = 500
BUFFER_M                 = 5000
ROI_FACTOR               = 2.0
INSIDE_STD_FACTOR        = 2            # How many mutiples of std to exclude within the radius. i.e. what is above background noise
BACKGROUND_STD_THRESHOLD = 0.02         # Standard deviation of the background must be smaller than 1.5 cm/yr
MIN_Q95_PIXELS           = 9           # For large landslides allow us get time series of fastest pixels
MIN_Q75_PIXELS           = 9           # Sets the miniumum resolveable size of landslide
SLOPE_THESHOLD           = 0 # needs fixing 2/90            # Minimum slope threhold. 
poly_order               = 4            # Polynomial order for calculating RMSE 
nn_scaled_threshold      = 2.25  
EDGE_THRESHOLD           = 0.8          # Fraction of valid data in radius at edge of the frame. 

GEOD = pyproj.Geod(ellps='WGS84')

eq1 = 2021.9685
eq2 = 2022.9685

wy1 = 2021.7479
wy2 = 2022.7479
wy3 = 2023.7479

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

SKIP_IDS = {
    "ls_4", "ls_67", "ls_69", "ls_71", "ls_73",
    "ls_1004", "ls_1013", "ls_1042", "ls_1045", "ls_1068",
    "ls_1073", "ls_1086", "ls_1093", "ls_1105", "ls_1117",
    "ls_1120", "ls_1121", "ls_1124", "ls_1125", "ls_1126",
    "ls_1133", "ls_1141", "ls_1144", "ls_1146", "ls_1165",
    "ls_1170", "ls_1182", "ls_1189", "ls_1201", "ls_1209",
    "ls_1211", "ls_1212", "ls_1213", "ls_1214", "ls_1220",
    "ls_1223", "ls_1230", "ls_1232", "ls_1235", "ls_1237",
    "ls_1238", "ls_1241", "ls_1244", "ls_1254", "ls_1255",
    "ls_1262", "ls_1266", "ls_1273", "ls_1280", "ls_1287",
    "ls_1288", "ls_1289", "ls_128", "ls_1290", "ls_1292",
    "ls_1296", "ls_1299", "ls_129", "ls_1302", "ls_1313",
    "ls_1329", "ls_1334", "ls_1357", "ls_1363", "ls_1364",
    "ls_1367", "ls_1368", "ls_1370", "ls_1377", "ls_1387",
    "ls_1401", "ls_1406", "ls_1413", "ls_1417", "ls_1419",
    "ls_1423", "ls_1428", "ls_1444", "ls_1449", "ls_1457",
    "ls_1460", "ls_1461", "ls_1462", "ls_1465", "ls_1467",
    "ls_1469", "ls_1473", "ls_1474", "ls_1475", "ls_1481",
    "ls_1487", "ls_1518", "ls_1524", "ls_1529", "ls_1542",
    "ls_1545", "ls_1551", "ls_1570", "ls_1573", "ls_1583",
    "ls_158", "ls_1596", "ls_184", "ls_186", "ls_218",
    "ls_224", "ls_226", "ls_233", "ls_234", "ls_237",
    "ls_241", "ls_247", "ls_251", "ls_253", "ls_256",
    "ls_263", "ls_266", "ls_267", "ls_270", "ls_271",
    "ls_272", "ls_280", "ls_298", "ls_300", "ls_301",
    "ls_302", "ls_304", "ls_305", "ls_306", "ls_307",
    "ls_312", "ls_313", "ls_315", "ls_316", "ls_320",
    "ls_322", "ls_341", "ls_349", "ls_357", "ls_359",
    "ls_365", "ls_366", "ls_368", "ls_416", "ls_437",
    "ls_445", "ls_447", "ls_448", "ls_449", "ls_452",
    "ls_454", "ls_456", "ls_457", "ls_459", "ls_461",
    "ls_462", "ls_465", "ls_472", "ls_473", "ls_477",
    "ls_499", "ls_503", "ls_510", "ls_521", "ls_541",
    "ls_551", "ls_552", "ls_602", "ls_606", "ls_611",
    "ls_623", "ls_635", "ls_643", "ls_649", "ls_679",
    "ls_686", "ls_690", "ls_691", "ls_699", "ls_700",
    "ls_714", "ls_717", "ls_719", "ls_727", "ls_731",
    "ls_756", "ls_766", "ls_789", "ls_791", "ls_792",
    "ls_793", "ls_810", "ls_812", "ls_829", "ls_896",
    "ls_900", "ls_915", "ls_925", "ls_926", "ls_927",
    "ls_928", "ls_930", "ls_938", "ls_940", "ls_942",
    "ls_948", "ls_954", "ls_955", "ls_958", "ls_965",
    "ls_973"
}

STATS = {k:0 for k in [
    'manual_skip','no_valid_data','high_background_std',
    'few_q75_pixels','high_nn_distance', 'low_slope','ts_q75', 'ts_q95', 'high_nn_scaled'
]}

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def calc_azi(lon1, lat1, lon2, lat2):
    """
    Compute azimuths and distance between two points on WGS84.

    Parameters
    ----------
    lon1, lat1 : float or array
        Origin longitude(s), latitude(s) in degrees.
    lon2, lat2 : float or array
        Destination longitude(s), latitude(s) in degrees.

    Returns
    -------
    az12 : float or array
        Forward azimuth from origin toward destination (° clockwise from North).
    az21 : float or array
        Back azimuth from destination back to origin (° clockwise from North).
    dist : float or array
        Geodesic distance between points in meters.
    """
    if np.isscalar(lon1):
        lon1 = np.full_like(lon2, lon1)
        lat1 = np.full_like(lat2, lat1)
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
    return radius, buffer_m, half_ext, x0, x1, y0, y1

def plot_roi_map(
    box_id, ls_id, ls_row, roi_df, inside, inside_ab, ls_q75, ls_q95, ls_sign,
    ts_df,
    radius, buffer, poly_gmt, fig_dir, cluster_label, coeffs_clean, rmse_clean,
    q75_value, q95_value, med
):
    fig_region = "%s/%s/%s/%s" % (np.nanmin(roi_df['Lon']), np.nanmax(roi_df['Lon']),
                                  np.nanmin(roi_df['Lat']), np.nanmax(roi_df['Lat']))

    bg_median = ls_row['ts_background_median']
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
             style="s0.09c", cmap=True)
    
    fig.plot(data=poly_gmt, pen="1.0p,black")
    
    fig.plot(x=ls_q75['Lon'], y=ls_q75['Lat'], fill="gold", style="c0.09c")
    
    if not ls_q95.empty:
        fig.plot(x=ls_q95['Lon'], y=ls_q95['Lat'],
                 fill="purple2", style="c0.09c")
    
    fig.plot(x=ls_row['Lon'], y=ls_row['Lat'], size=[
             radius*2/1000], style="E-", pen="1.5p,darkorange", transparency=50)
    fig.plot(x=ls_row['Lon'], y=ls_row['Lat'], size=[
             buffer*2/1000], style="E-", pen="1.5p,dodgerblue4,--", transparency=50)

    fig.text(text="%s" % ls_row['ID'], position="TC",
             offset="0.0c/-0.2c", font="14p,Helvetica-Bold")

    fig.basemap(region=fig_region, projection="M5.9c", frame=["tSWr", 'xa0.04', 'ya0.02'],
                map_scale="jTR+w1k+o0.4/0.4c",)

    fig.colorbar(cmap=True, frame=["x", "y+lm/yr"],
                 position="JBC+o0c/0.75c+w4c/0.4c+h")

    fig.shift_origin(xshift="7.5c")
    
    fig.histogram(data=inside["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                  frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="gold", transparency=50, label='inside radius')
    q75_x = med*100 + q75_value*ls_sign*100
    q95_x = med*100 + q95_value*ls_sign*100
    
    fig.histogram(data=inside_ab["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                  frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="orange", transparency=50, label='exluding location med+/-*2std')
    fig.histogram(data=ls_q75["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                  frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="red", transparency=50, label='q75')
    if not ls_q95.empty:
        fig.histogram(data=ls_q95["Vel"]*100, projection="X6c/4c", region=[vmin, vmax, 0, count_max], series=[vmin, vmax, 0.5],
                      frame=["WtSr", "xaf+lLOS Velocity (cm/yr)", "yaf+lCounts"], histtype=0, pen="1p,black", fill="purple2", transparency=50, label='q95')
    fig.plot(x=[q75_x, q75_x], y=[0, count_max],  pen="1p,red,--", label="q75 value")
    fig.plot(x=[q95_x, q95_x], y=[0, count_max],  pen="1p,purple2,--", label="q95 value")
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
        ls_row['ts_background_std']*100, 1)), position="TL", offset="0.2c/-0.7c", font="Helvetica")
    
    fig.text(text="Dist. to ref %s km" % (np.round(
        ls_row['ref_dist']/1000, 1)), position="TL", offset="0.2c/-1.2c", font="Helvetica")


    fig.text(text="Pixels Q75: %s, Q95: %s" % (len(ls_q75), len(ls_q95)),
             position="TL", offset="0.2c/-1.7c", font="Helvetica")
    
    fig.text(text="NN Scaled %s" % (np.round(
        ls_row['ts_nn_scaled'], 2)), position="TL", offset="0.2c/-2.2c", font="Helvetica")
    
    # Right text 
    fig.text(text="Radius: %s km" % (np.round(radius/1000, 1)),
             position="TL", offset="3c/-0.2c", font="Helvetica")
    
    fig.text(text="TS source: %s " % (cluster_label),
             position="TL", offset="3c/-0.7c", font="Helvetica")
    
    fig.text(text="NN dist: %s " % (np.round(
        ls_row['ts_mean_nn'], 1)),
             position="TL", offset="3c/-2.2c", font="Helvetica")

    fig.shift_origin(yshift="-4.5c", xshift="7c")
    region_ts = [np.nanmin(ts_df["dates"])-0.2, np.nanmax(ts_df["dates"])+0.2, np.nanmin(ts_df['clean_ts'])-0.02, np.nanmax(ts_df['clean_ts'])+0.02]
    fig.basemap(region=region_ts, projection="X12c/7.5c", frame=["lStE", "xaf+lYears", "yaf+lLOS Displacement (m)"])

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

    # 1) Linear fit (dates in years, clean_ts in meters)
    slope, intercept, _, _, _ = linregress(ts_df["dates"], ts_df["clean_ts"])
    
    # 2) compute predicted values and residuals
    y_pred    = intercept + slope * ts_df["dates"]
    residuals = ts_df["clean_ts"] - y_pred
    
    # 3) RMSE of the linear model
    rmse_lin = np.sqrt(np.mean(residuals**2))    # m (or same units as y)

    # 1) make a nice dense set of x’s between min and max date
    x_fit = np.linspace(ts_df["dates"].min(), ts_df["dates"].max(), 200)
    
    # 2) evaluate the polynomial at those x’s
    #    coeffs_clean is like [a, b, c] from np.polyfit(...)
    y_fit = np.polyval(coeffs_clean, x_fit)
    
    fig.text(text=f"Lin. {np.round(slope * 100,1)} RMSE {np.round(rmse_lin*100,1)} cm/yr, Poly. RMSE {np.round(rmse_clean*100,1)} cm/yr",
             position="BC", offset="0/0.2c", font="Helvetica")
    
    fig.plot(x=ts_df["dates"], y= intercept + slope * ts_df["dates"],pen="2p,dodgerblue4,--", label="linear fit", transparency=50)
    
    # 3) plot the line
    fig.plot(x=x_fit, y=y_fit, pen="2p,dodgerblue4", label=f"{poly_order} order fit", transparency=50)
    
    fig.plot(x=ts_df["dates"], y=ts_df["median_ls"], style="c0.15c",
             pen="1p,darkorange", fill="darkorange", label="Original")
    
    fig.plot(x=ts_df["dates"], y=ts_df["clean_ts"], style="c0.15c",
             pen="1p,dodgerblue4", fill="dodgerblue4", label="Cleaned")
    fig.legend()
    
    fname = f"{fig_dir}/All_Landslide_TS/{ls_row['ID']}_{box_id}_radius{np.round(radius,1)}_buffer{buffer}_{cluster_label}_{dataset_label}.png"
    fig.savefig(fname, transparent=False, crop=True,
                anti_alias=True, show=False)
    #fig.show()
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

def total_area_m2_numPixels(x_step_deg, y_step_deg, num_pixels, lon0_deg, lat0_deg):
    """
    Compute the total area (in m²) for a given number of pixels, where each pixel
    has size x_step_deg × y_step_deg in geographic degrees.

    Parameters
    ----------
    x_step_deg : float
        Pixel width in degrees longitude.
    y_step_deg : float
        Pixel height in degrees latitude.
    num_pixels : int
        Total number of pixels.
    lon0_deg : float
        Reference longitude (center of pixel grid).
    lat0_deg : float
        Reference latitude (center of pixel grid).
    Returns
    -------
    float
        Total area in square meters.
    """
    geod = Geod(ellps="WGS84")
    
    # Cast inputs to native Python types
    x_step = float(x_step_deg)
    y_step = float(y_step_deg)
    lon0   = float(lon0_deg)
    lat0   = float(lat0_deg)
    n_pix  = int(num_pixels)

    geod = Geod(ellps="WGS84")

    # East-west meter distance for one pixel
    _, _, dx = geod.inv(lon0, lat0, lon0 + x_step, lat0)
    # North-south meter distance for one pixel
    _, _, dy = geod.inv(lon0, lat0, lon0, lat0 + y_step)

    return abs(dx * dy) * n_pix

def check_cluster(cand, label, x_step, y_step):
    """
    Evaluate whether a candidate pixel cluster is both large enough
    and spatially compact enough to accept.

    Parameters
    ----------
    cand : pandas.DataFrame
        Subset of inside_ab pixels (with 'Lon' and 'Lat' columns).
    label : str
        Either "q95" or "q75", used to pick the right pixel‐count threshold.

    Returns
    -------
    dict or None
        If the cluster meets the minimum‐pixel and packing‐density tests,
        returns a dict with keys:
          - "cluster":   the DataFrame of pixels
          - "label":     the cluster label ("q95" or "q75")
          - "mean_nn":   the mean nearest‐neighbor distance (m)
          - "scaled_nn": mean_nn / ideal_nn (unitless)
        Otherwise returns None.
    """
    Npix = len(cand)

    # Compute approximate per‐pixel area in m²
    cluster_lat  = cand["Lat"].mean()
    cluster_lon  = cand["Lon"].mean()
    cluster_area = total_area_m2_numPixels(
        x_step, y_step, Npix, cluster_lon, cluster_lat)
    pixel_area = cluster_area / Npix

    # 3) Ideal nearest‐neighbor spacing for perfect packing
    ideal_nn = np.sqrt(pixel_area)

    # 4) Compute actual mean NN: for each pixel, find its closest neighbor
    coords   = cand[["Lon","Lat"]].values
    nn_dists = []
    for i, (lon_i, lat_i) in enumerate(coords):
        other = np.delete(coords, i, axis=0)
        _, _, d = calc_azi(lon_i, lat_i, other[:,0], other[:,1])
        nn_dists.append(d.min())
    mean_nn = np.mean(nn_dists)

    # 5) Scale observed NN by the ideal one (dimensionless)
    scaled_nn = mean_nn / ideal_nn

    # 6) Return all the metrics if accepted
    return mean_nn, scaled_nn

# -------------------------------------------------------------------------
# Per‐landslide processing
# -------------------------------------------------------------------------
def process_landslide(ls_row, vel_lons, vel_lats, lon_1d, lat_1d,
                      ncols, nrows, ref_lon, ref_lat,
                      vel_file, ts_file, fig_dir,
                      clean_df, err_low_df, err_high_df,
                      box_id):
    
    ls_row['insar_dataset'] = dataset_label
    ls_row['insar_box_id'] = box_id
    
    # Assign location lat and lon values
    ls_id, ls_lon, ls_lat = ls_row['ID'], ls_row['Lon'], ls_row['Lat']
    print(f"\n→ {ls_id}")
    
    
    # 1) Calculate distances
    _,_,ref_d = calc_azi(ref_lon, ref_lat, ls_lon, ls_lat)
    eq1_azi2ls,_,eq1_dist = calc_azi(eq_21_lon, eq_21_lat, ls_lon, ls_lat)
    eq2_azi2ls,_,eq2_dist = calc_azi(eq_22_lon, eq_22_lat, ls_lon, ls_lat)
    ls_row['ref_dist'] = ref_d
    ls_row['eq_21_dist'] = eq1_dist
    ls_row['eq_21_azi2ls'] = eq1_azi2ls
    ls_row['eq_22_dist'] = eq2_dist
    ls_row['eq_22_azi2ls'] = eq2_azi2ls
    # ------------------------------
    # Get ROI
    # ------------------------------
    
    # raw ROI window, calculate raidus based on the _maximum diameter of the landslide. 
    r, b, half_ext, x0, x1, y0, y1 = compute_roi_window(
        ls_lon, ls_lat, lon_1d, lat_1d, ncols, nrows,
        ls_row['ls_max_diameter_m']
    )
    
    # Set up ROI window
    xs, xe = sorted((x0,x1)); ys, ye = sorted((y0,y1))

    # read velocity block from hdf5
    roi_lons = vel_lons[ys:ye, xs:xe]
    roi_lats = vel_lats[ys:ye, xs:xe]
    with h5py.File(vel_file,'r') as hf:
        vel = hf['velocity'][ys:ye, xs:xe]
        x_step = float(hf.attrs['X_STEP'])
        y_step = float(hf.attrs['Y_STEP'])
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
        ls_row['reject_reason'] = "no_valid_data"
        print("   • reject: no valid data")
        return ls_row
    
    # ------------------------------
    # Calculate background Std
    # ------------------------------
    
    # Calcualte background stats. reject if background noise is too high
    bg = df[df['distance']>b]
    bg_med, bg_std = np.nanmedian(bg['Vel']), np.nanstd(bg['Vel'])
    ls_row['ts_background_std'] = bg_std
    ls_row['ts_background_median'] = bg_med
    
    # ------------------------------
    # Set landslide sign, q75 and q95 values. 
    # ------------------------------
        
    # Get all data within the radius value. 
    inside = df[df['distance']<r]
        
    # Keep only pixels above background std * 2
    inside_ab = inside[
        (inside['Vel'] > bg_med + INSIDE_STD_FACTOR * bg_std) |
        (inside['Vel'] < bg_med - INSIDE_STD_FACTOR * bg_std)
    ]
        
    # 1) compute pixels at 75% & 95%. Landslide sign is whether it is postive and negative
    ls_sign = np.sign(np.nanmedian(inside_ab['Vel'] - bg_med))
    ls_row['ls_sign'] = ls_sign
    
    # Make absolute to calculate the quartiles. 
    ls_absolute = (inside_ab['Vel'] - bg_med) * ls_sign

    # Set q75 velocity threshold
    q75_value = np.nanquantile(ls_absolute, 0.75)
    ls_q75 = inside_ab[ls_absolute > q75_value]
    ls_row['ts_num_q75'] = len(ls_q75)
    
    # Set q95 velocity threshold
    q95_value = np.nanquantile(ls_absolute, 0.95)
    ls_q95 = inside_ab[ls_absolute > q95_value]
    ls_row['ts_num_q95'] = len(ls_q95)
    
    # Decide which cluster to use based on the number of pixels. Reject if not enough q75 pixels
    if len(ls_q95) >= MIN_Q95_PIXELS:
        ls_cluster   = ls_q95
        cluster_label = "q95"
        print(f"   • using Q95 cluster ({len(ls_q95)} pixels)")
    else:
        ls_cluster    = ls_q75
        cluster_label = "q75"     
        print(f"   • using Q75 cluster ({len(ls_q75)} pixels)")
    
    if len(ls_q75) == 0:
        ls_row['reject_reason'] = "no_q75_pixels"
        print(f"   • reject: no Q75 pixels (0); skipping")
        return ls_row
    
    # ------------------------------
    # Calculate Statistics
    # ------------------------------
    try:        
        # Calculate geometry metrics
        geo_file = os.path.join(box, "geo", "geo_geometryRadar.h5")
        geo_means = extract_geometry_means(geo_file, ys, ye, xs, xe, ls_cluster)
        mean_slope = geo_means.get('slope',  np.nan)
        ls_row['ls_mean_height'] = geo_means.get('height', np.nan),
        ls_row['ls_mean_slope'] = geo_means.get('slope', np.nan),
        ls_row['ls_mean_aspect'] = geo_means.get('aspect', np.nan),
        ls_row['ls_mean_incidenceAngle'] = geo_means.get('incidenceAngle', np.nan)
        
        cluster_lat = np.mean(ls_cluster[['Lat']].values)
        cluster_lon = np.mean(ls_cluster[['Lon']].values)
        cluster_area_m2 = total_area_m2_numPixels(x_step, y_step, len(ls_cluster), cluster_lon, cluster_lat)
        ls_row['ts_cluster_lat'] = cluster_lat
        ls_row['ts_cluster_lon'] = cluster_lon
        ls_row['ts_cluster_area_m2'] = cluster_area_m2
    
        # Checking for edge effect, calculated expecte number of pixels within the given radius and then fraction of valid data
        # Get per pixel area
        geod = Geod(ellps="WGS84")
        _,_,dx = geod.inv(ls_lon, ls_lat, ls_lon + x_step, ls_lat)
        _,_,dy = geod.inv(ls_lon, ls_lat, ls_lon, ls_lat + y_step)
        pixel_area = abs(dx * dy)  # m² per pixel
    
        # compute expected count in a full circle
        circle_area = math.pi * (r**2)  # m²
        expected_n = circle_area / pixel_area
    
        # now your existing counts
        mask   = df['distance'] < r
        n_tot  = mask.sum()
        fraction_data = n_tot / expected_n
            
        # 4) Pixel clustering 
        if cluster_label == "q95":
            mean_nn, scaled_nn = check_cluster(ls_q95, "q95", x_step, y_step)
            if scaled_nn > nn_scaled_threshold:
                print("   • Q95 cluster failed NN test; falling back to Q75")
                mean_nn, scaled_nn = check_cluster(ls_q75, "q75", x_step, y_step)
                cluster_label = "q75"
                ls_cluster = ls_q75
            else: 
                print(f"   • accepted Q95 cluster  (scaled_nn={scaled_nn:.2f})")
        else:
            # we either started as q75, or we fell back from q95
            mean_nn, scaled_nn = check_cluster(ls_q75, "q75", x_step, y_step)
        
        ls_row['ts_mean_nn'] = mean_nn
        ls_row['ts_nn_scaled'] = scaled_nn
                
    except:
        # if *any* exception is raised above, this block runs
        print("Couldn't get stats")
        
    # Reject based on number of pixels.
    if len(ls_q75) < MIN_Q75_PIXELS:
        ls_row['reject_reason'] = "few_q75_pixels"
        print(f"Q75 cluster ({len(ls_q75)} pixels) --> too few Q75 pixels; skipping")
        return ls_row
    # ------------------------------
    # Reject based on thresholds
    # ------------------------------
    
    # Reject based location at edge of InSAR frame
    if (n_tot == 0 or fraction_data < EDGE_THRESHOLD):
        ls_row['reject_reason'] = "edge_of_frame"
        print(f"   • reject: {n_tot}/{expected_n} valid, {n_tot/expected_n:.2f} of circle")
        return ls_row

    # Reject based on standard deviation.
    if bg_std > BACKGROUND_STD_THRESHOLD:
        ls_row['reject_reason'] = "high_background_std"
        print(f"   • high bg std {np.round(bg_std,2)}")
        return ls_row

    # Reject based on nn_scaling.
    if scaled_nn > nn_scaled_threshold:
        ls_row['reject_reason'] = "high_nn_scaled"
        print(f"   • Q75 cluster failed NN test {np.round(scaled_nn,2)}")
        return ls_row

    # # Reject if slope is less than slope threshold
    # if mean_slope < SLOPE_THESHOLD:
    #     ls_row['reject_reason'] = "low_slope"
    #     print(f"   • Slope {np.round(mean_slope),1} below threshold; skipping")
    #     return ls_row

    # Check landslide is not in a manual skip list
    if ls_id in SKIP_IDS:
        ls_row['reject_reason'] = "manual_skip"
        print("   • Manual skipped")
        return ls_row

    # Sucess! Time to extract the time series. 
    ls_row['reject_reason'] = "Success"
    ls_row['ts_cluster_label'] = cluster_label
    ls_row['ts_num_cluster'] = len(ls_cluster)
        
    # ------------------------------
    # Extract Timeseries 
    # ------------------------------
    
    # Extract time series for the landslide
    ts_df, clean_df, err_low_df, err_hi_df = extract_and_plot_timeseries(
        box_id, ls_id, ls_row, ls_cluster, roi_lons, roi_lats,
        b, r, ts_file, fig_dir, ys, ye, xs, xe,
        clean_df, err_low_df, err_high_df
    )
    
    # 4.a) compute goodness‐of‐fit metrics
    coeffs_clean, rmse_clean = poly_rmse(ts_df["dates"].values,
                                         ts_df["clean_ts"].values,
                                         deg=poly_order)
    
    #    optionally also on the raw median LS:
    coeffs_orig, rmse_orig = poly_rmse(ts_df["dates"].values,
                                       ts_df["median_ls"].values,
                                       deg=poly_order)
    
    # percentage of RMSE improved with cleaning    
    ls_row['ts_pct_rmse_red'] = 100 * (rmse_orig - rmse_clean) / rmse_orig
    
    # Set parameters into ls_row for saving metadata. 
    ls_row['ts_rmse_clean_m'] = rmse_clean,
    ls_row['ts_rmse_orig_m'] = rmse_orig,
    ls_row['ts_poly_deg'] = poly_order,
    ls_row['ts_poly_coeffs_clean'] = coeffs_clean.tolist(),
    ls_row['ts_poly_coeffs_orig'] = coeffs_orig.tolist(),
    
    # ------------------------------
    # Extract Timeseries 
    # ------------------------------
    
    # Whole dataset 
    ls_row['ts_linear_vel_myr'], ls_row['ts_linear_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=np.nanmin(ts_df["dates"].values), stop=np.nanmax(ts_df["dates"].values))
    
    # Dry Months Only
    ls_row['ts_dry1_vel_myr'], ls_row['ts_dry1_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=vel_t1, stop=vel_t2)
    ls_row['ts_dry2_vel_myr'], ls_row['ts_dry2_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=vel_t3, stop=vel_t4)
    
    # Post EQ 3 months 
    ls_row['ts_eq1_3month_vel_myr'], ls_row['ts_eq1_3month_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=eq1, stop=eq1+3/12)
    ls_row['ts_eq2_3month_vel_myr'], ls_row['ts_eq2_3month_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=eq2, stop=eq2+3/12)
    
    # Post EQ 6 months 
    ls_row['ts_eq1_6month_vel_myr'], ls_row['ts_eq1_6month_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=eq1, stop=eq1+6/12)
    ls_row['ts_eq2_6month_vel_myr'], ls_row['ts_eq2_6month_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=eq2, stop=eq2+6/12)
    
    # Post EQ 6 months 
    ls_row['ts_eq1_12month_vel_myr'], ls_row['ts_eq1_12month_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=eq1, stop=eq1+12/12)
    ls_row['ts_eq2_12month_vel_myr'], ls_row['ts_eq2_12month_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=eq2, stop=eq2+12/12)
    
    # Water Years 
    ls_row['ts_wy22_vel_myr'], ls_row['ts_wy22_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=wy1, stop=wy2)
    ls_row['ts_wy23_vel_myr'], ls_row['ts_wy23_err_myr'] = compute_velocity(ts_df["dates"].values, ts_df["clean_ts"].values, start=wy2, stop=wy3)

    #ls_row['ts_pct_rmse_red'] = pct_rmse_reduced,
    # ls_row['ts_linear_vel_myr'] = linear_vel,
    # ls_row['ts_linear_err_myr'] = linear_err,
    # ls_row['ts_dry1_vel_myr'] = vel_dry1,
    # ls_row['ts_dry1_err_myr'] = err_dry1,
    # ls_row['ts_dry2_vel_myr'] = vel_dry2,
    # ls_row['ts_dry2_err_myr'] = err_dry2,
    
    # save dataframes as array for output into hdf5
    dates_arr    = ts_df["dates"].values                 # shape (N,)
    clean_arr    = clean_df[ls_id].values                # shape (N,)
    err_low_arr  = err_low_df[ls_id].values              # shape (N,)
    err_high_arr = err_high_df[ls_id].values             # shape (N,)
    orginal_arr  = ts_df["median_ls"].values            # shape (N,)
    
     
    # # Plot time series 
    # plot_roi_map(
    #    box_id, ls_id, ls_row, df, inside, inside_ab, ls_q75, ls_q95, ls_sign,
    #    ts_df,
    #    r, b, poly_gmt, fig_dir, cluster_label, coeffs_clean, rmse_clean, 
    #    q75_value, q95_value, bg_med, 
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
            'radius_used_m':            r,    # Radius to grab pixels within. 
            'buffer_m':                 b,    # Radius for background
            'half_roi_extend':          half_ext,   # half_ext = buffer_m * ROI_FACTOR
            'default_radius':           DEFAULT_RADIUS_M,  # Radius for the circle surrounding the landslide
            'default_buffer':           BUFFER_M,      # Distance between inner circle and outter
            'default_roi_factor':       ROI_FACTOR,  # ROI for background, mutiple of buffer_m 
            'threshold_ins_std_factor': INSIDE_STD_FACTOR, # How many mutiples of std to exclude within the radius. i.e. what is above background noise
            'threshold_bg_std':         BACKGROUND_STD_THRESHOLD,   # How many mutiples of std to exclude within the radius. i.e. what is above background noise
            'threshold_min_num_q95':    MIN_Q95_PIXELS, # For large landslides allow us get time series of fastest pixels
            'threshold_min_num_q75':    MIN_Q75_PIXELS, # Sets the miniumum resolveable size of landslide
            'threhold_nn_ratio':        nn_scaled_threshold,
            'threshold_slope':          SLOPE_THESHOLD,
            'threshold_edge_data':      EDGE_THRESHOLD,
            'insar_dataset':            dataset_label,
            'eq21_date':                eq1, 
            'eq21_lon':                 eq_21_lon, 
            'eq21_lat':                 eq_21_lat, 
            'eq22_date':                eq2, 
            'eq22_lon':                 eq_22_lon, 
            'eq22_lat':                 eq_22_lat,
            'dry_21_start':             vel_t1,
            'dry_21_end':               vel_t2,
            'dry_22_start':             vel_t3,
            'dry_22_end':               vel_t4,
            
        }.items():
            p.attrs[k] = v

        # Add all columns in ls_row as attributes under 'meta' group        
        slide_info = ls_row.to_dict()
        m = hf.create_group("meta")
        m.attrs.update(slide_info)
            
        # Now automatically pick up _orig_id columns:
        for col, val in ls.items():
            if col.endswith('_orig_id'):
                # skip NaN or empty strings
                if isinstance(val, str) and val.strip():  
                    slide_info[col] = val
                elif not isinstance(val, str) and not pd.isnull(val):
                    slide_info[col] = val
        
        # Write them all into HDF5 attrs:
        for key, value in slide_info.items():
            m.attrs[key] = value
            
        # 3) time‐series at root
        hf.create_dataset("dates",     data=dates_arr,    dtype='f8')
        hf.create_dataset("clean_ts",  data=clean_arr,    dtype='f8')
        hf.create_dataset("err_low",   data=err_low_arr,  dtype='f8')
        hf.create_dataset("err_high",  data=err_high_arr, dtype='f8')
        hf.create_dataset("ori_ts",    data=orginal_arr,  dtype='f8')

    print(f"✔  wrote {out_h5}")
    return ls_row

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__=="__main__":
    root     = ts_dir
    poly_csv = common_paths['ls_inventory']
    poly_gmt = common_paths['ls_gmt']['all_polygons']
    fig_dir  = fig_dir
    data_dir = ts_out_dir
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ts_out_dir, exist_ok=True)

    # load landslide metrics
    slides = pd.read_csv(poly_csv, dtype={'ls_id':str})
    slides.rename(columns={'ls_id':'ID','center_lon':'Lon','center_lat':'Lat'}, inplace=True)

    #boxes = sorted(glob.glob(os.path.join(root,"y*_box")))
    boxes = sorted(glob.glob(os.path.join(root,"y*_box")))

    meta_data = []
    
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

        
        
        for _, ls in subset.iterrows():
            rec = process_landslide(
                ls, vel_lons, vel_lats, lon_1d, lat_1d,
                ncols, nrows, ref_lon, ref_lat,
                vel_file, ts_file, fig_dir,
                clean_df, err_low_df, err_hi_df,
                box_id
            )
            if rec is not None:
                meta_data.append(rec)

        # after all boxes are done, write the summary once:
        df_meta   = pd.DataFrame(meta_data)
        out_meta  = os.path.join(data_dir, f"processing_summary_{dataset_label}_{box_id}.csv")
        df_meta.to_csv(out_meta, index=False)
        print(f"Full processing summary (with reject reasons) saved to {out_meta}")


