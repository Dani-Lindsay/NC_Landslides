#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:45:17 2025

@author: andrewmcnab
"""

import h5py
import pandas as pd
import datetime
# from .rainfall_prism import prism_cumulative_rainfall
# from .geology_macrostrat_usgs import get_geology
from pga_shakemap import get_shakemap_df

def minmax_date(landslide_data):
    all_timestamps = [
        t
        for key in landslide_data
        for t in landslide_data[key]['ts']['timestamps']
    ]
    unique_timestamps = list(set(all_timestamps))
    min_ts = min(unique_timestamps)
    max_ts = max(unique_timestamps)
    min_rounded_back = (min_ts.normalize())
    max_rounded_forward = (max_ts.normalize())
    min_date_str = min_rounded_back.strftime("%Y-%m-%d")
    max_date_str = max_rounded_forward.strftime("%Y-%m-%d")
    
    return min_date_str, max_date_str

def decimal_year_to_datetime(dec_year):
    s = str(dec_year)
    if '.' in s:
        year_str, frac_str = s.split('.', 1)
    else:
        year_str, frac_str = s, '0'

    year = int(year_str)
    frac = int(frac_str) / (10 ** len(frac_str))

    start = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
    end   = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    return start + (end - start) * frac

def load_landslide_ts(h5_path):
    """
    Reads an HDF5 landslide time‐series file and returns:
     - landslide_data: dict mapping landslide ID to a DataFrame with:
         • index: decimal‐year dates
         • columns: 'los', 'err_low', 'err_high', 'lat', 'lon'
     - attrs: dict of global file attributes (processing parameters)
    """
    with h5py.File(h5_path, 'r') as f:
        # Read arrays
        dates = f['dates'][:]  # shape (n_epochs,)
        ls_ids = [s.decode('ascii').strip() for s in f['ls_id'][:]]
        lats = f['ls_lat'][:]
        lons = f['ls_lon'][:]
        clean_ts = f['clean_ts'][:, :]      # (n_epochs, n_slides)
        err_low = f['err_low'][:, :]
        err_high = f['err_high'][:, :]
        
        # Read global attributes
        attrs = {key: f.attrs[key] for key in f.attrs}

    # Build a DataFrame per landslide
    landslide_data = {}
    for i, ls in enumerate(ls_ids):
        timestamps = [decimal_year_to_datetime(d) for d in dates]
        df = pd.DataFrame({
            'decimal_year': dates,
            'timestamps': timestamps,
            'los': clean_ts[:, i],
            'err_low': err_low[:, i],
            'err_high': err_high[:, i],
            'lat': lats[i],
            'lon': lons[i],
        })
        #df.set_index('date', inplace=True)
        landslide_data.setdefault(ls, {})['ts'] = df
    
    min_date_str, max_date_str = minmax_date(landslide_data)
    
    attrs['global_start_date'] = min_date_str
    attrs['global_end_date'] = max_date_str

    return landslide_data, attrs

def landslide_locations(landslide_data):
    landslidelocations = []
    for landslide_id in landslide_data.keys():
        df = landslide_data[landslide_id]['ts']
        d = {
            "site_id": landslide_id,
            "lon": float(df.lon.unique()[0]),
            "lat": float(df.lat.unique()[0]),
            }
        landslidelocations.append(d)
    
    return landslidelocations
    
#pga, pgv, mmi per event (directivity)
#search for other events to get get cumlative pga, pgv, mmi
#rainfall
#lithology
#coastal flag
#road flag
#elevation





