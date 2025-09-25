#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augments displacement HDF5 with supporting attributes and resamples supporting
data to match a 14-day interval starting from the first InSAR acquisition date.

- Retains existing meta and params groups
- Adds support_meta and support_params
- Adds dates_14day and dates_14day_decimal
- Adds new groups: geology, pga, rainfall

@author: daniellelindsay
"""

import h5py
import numpy as np
import pandas as pd
from datetime import timedelta
import os
from hdf5_support import load_landslide_hdf5, _save_item
from landslide_utlities import date_to_decimal_year, decimal_year_to_datetime

# -------- File paths --------
displacement_file = "/Volumes/Seagate/NC_Landslides/test_data/ls_001_y3_x1_box_q95_Timeseries_1.h5"
supporting_file = "/Volumes/Seagate/NC_Landslides/test_data/ls_001-supporting.h5"

# -------- Load supporting data --------
support_data = load_landslide_hdf5(supporting_file)

with h5py.File(displacement_file, 'r+') as disp:

    # -------- Handle dates from file --------
    raw_dates = disp["dates"][:]
    if np.issubdtype(raw_dates.dtype, np.number):
        insar_original_dates = decimal_year_to_datetime(raw_dates)
    elif np.issubdtype(raw_dates.dtype, np.bytes_):
        insar_original_dates = pd.to_datetime([d.decode("utf-8") for d in raw_dates])
    else:
        insar_original_dates = pd.to_datetime(raw_dates)

    anchor_date = insar_original_dates.min()
    end_date = insar_original_dates.max()

    # -------- Generate 14-day cadence --------
    insar_14day_dates = pd.date_range(start=anchor_date, end=end_date, freq="14D")

    # Save as /dates_14day (UTF-8)
    encoded_14day_dates = np.array([d.strftime("%Y%m%d").encode("utf-8") for d in insar_14day_dates])
    if "dates_14day" in disp:
        del disp["dates_14day"]
    disp.create_dataset("dates_14day", data=encoded_14day_dates)

    # Save as /dates_14day_decimal
    decimals = np.array([date_to_decimal_year(d) for d in insar_14day_dates])
    if "dates_14day_decimal" in disp:
        del disp["dates_14day_decimal"]
    disp.create_dataset("dates_14day_decimal", data=decimals)

    # Use new 14-day dates for sampling
    insar_dates = insar_14day_dates

    # -------- Add support_meta group --------
    if "support_meta" in disp:
        del disp["support_meta"]
    _save_item(disp, "support_meta", {
        "eq_event_radius": support_data["meta"]["eq_event_radius"]
    })

    # -------- Add support_params group --------
    if "support_params" in disp:
        del disp["support_params"]
    _save_item(disp, "support_params", {
        "dist_ocean_m": support_data["distances"]["ocean_m"],
        "dist_road_m": support_data["distances"]["road_m"]
    })

    # -------- /geology --------
    if "geology" in disp:
        del disp["geology"]
    _save_item(disp, "geology", support_data["geology"])

    # -------- /pga --------
    if "pga" in disp:
        del disp["pga"]
    _save_item(disp, "pga", support_data["pga"])

    # -------- /rainfall --------
    if "rainfall" in disp:
        del disp["rainfall"]
    _save_item(disp, "rainfall", support_data["rainfall"])

    # -------- Resample rainfall --------
    rain_dates = pd.to_datetime(support_data["rainfall"]["date"]).tz_localize(None)
    rain_mm = np.array(support_data["rainfall"]["rain_mm"])
    rain_mm_cum = []
    rain_mm_peak = []

    for t in insar_dates:
        mask = (rain_dates >= t) & (rain_dates < t + timedelta(days=14))
        rain_mm_cum.append(np.nansum(rain_mm[mask]) if np.any(mask) else np.nan)
        rain_mm_peak.append(np.nanmax(rain_mm[mask]) if np.any(mask) else np.nan)

    for name, data in {
        "rain_mm_cum": rain_mm_cum,
        "rain_mm_peak": rain_mm_peak
    }.items():
        if name in disp["rainfall"]:
            del disp["rainfall"][name]
        disp["rainfall"][name] = np.array(data)

    # -------- Resample PGA --------
    pga_time = pd.to_datetime(support_data["pga"]["event_time"]).tz_localize(None)
    pga_mean = np.array(support_data["pga"]["pga_mean"])
    pgv_mean = np.array(support_data["pga"]["pgv_mean"])
    threshold = 0.001  # PGA threshold

    pga_mean_cum = []
    pga_mean_peak = []
    pgv_mean_cum = []
    pgv_mean_peak = []
    eq_count = []

    for t in insar_dates:
        mask = (pga_time >= t) & (pga_time < t + timedelta(days=14))
        if np.any(mask):
            pga_mean_cum.append(np.nansum(pga_mean[mask]))
            pga_mean_peak.append(np.nanmax(pga_mean[mask]))
            pgv_mean_cum.append(np.nansum(pgv_mean[mask]))
            pgv_mean_peak.append(np.nanmax(pgv_mean[mask]))
            eq_count.append(np.sum(pga_mean[mask] > threshold))
        else:
            pga_mean_cum.append(np.nan)
            pga_mean_peak.append(np.nan)
            pgv_mean_cum.append(np.nan)
            pgv_mean_peak.append(np.nan)
            eq_count.append(0)

    for name, data in {
        "pga_mean_cum": pga_mean_cum,
        "pga_mean_peak": pga_mean_peak,
        "pgv_mean_cum": pgv_mean_cum,
        "pgv_mean_peak": pgv_mean_peak,
        "eq_count": eq_count,
    }.items():
        if name in disp["pga"]:
            del disp["pga"][name]
        disp["pga"][name] = np.array(data)

print("✅ Done updating:", os.path.basename(displacement_file))
