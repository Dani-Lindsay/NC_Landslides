#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 19:19:03 2025

@author: daniellelindsay
"""

from hdf5_support import load_landslide_hdf5, _save_item
import os
import pandas as pd
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from NC_Landslides_paths import *


# This is the correct ls_id - target
displacement_file = os.path.join(ts_final_dir, "final_selection_only.csv")

# These are the old ones
supporting_dir   = "/Volumes/Seagate/NC_Landslides/Inputs/landslide_supporting"

def _pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing any of columns: {candidates}")
    return None

def _latlon_to_xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    clat = np.cos(lat)
    return np.column_stack((clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)))

def map_target_to_support(df_support, df_target, earth_radius_m=6371008.8):
    # Column resolution
    s_lon = _pick_col(df_support, ["lon", "Lon", "LON"])
    s_lat = _pick_col(df_support, ["lat", "Lat", "LAT"])
    s_id  = _pick_col(df_support, ["event_id", "id", "event"])

    t_lon = _pick_col(df_target, ["center_lon", "lon", "Lon", "LON", "longitude", "Longitude"])
    t_lat = _pick_col(df_target, ["center_lat", "lat", "Lat", "LAT", "latitude", "Latitude"])
    t_id  = _pick_col(df_target, ["ls_id", "id", "LS_ID"])
    
    # Keep only finite rows
    sup = df_support[[s_id, s_lon, s_lat]].copy()
    sup = sup[np.isfinite(sup[s_lon]) & np.isfinite(sup[s_lat])]
    if len(sup) == 0:
        raise ValueError("No valid rows in df_support with finite lon/lat.")

    tar = df_target.copy()
    tar = tar[np.isfinite(tar[t_lon]) & np.isfinite(tar[t_lat])]
    if len(tar) == 0:
        raise ValueError("No valid rows in df_target with finite center_lon/center_lat.")

    ("Starting KDTree on Support")
    # KDTree on support (parents are target → we query supports)
    sup_xyz = _latlon_to_xyz(sup[s_lat].to_numpy(), sup[s_lon].to_numpy())
    tree = cKDTree(sup_xyz)

    ("Query nearest support")
    # Query nearest support for each target
    tar_xyz = _latlon_to_xyz(tar[t_lat].to_numpy(), tar[t_lon].to_numpy())
    d_chord, idx = tree.query(tar_xyz, k=1)

    # Convert chord length to great-circle distance (meters)
    theta = 2.0 * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))
    dist_m = earth_radius_m * theta

    ("Build Output")
    # Build output aligned to tar (all targets preserved)
    sup_near = sup.iloc[idx].reset_index(drop=True)
    out = tar[[t_id, t_lon, t_lat]].reset_index(drop=True).copy()
    out.rename(columns={t_id: "ls_id", t_lon: "center_lon", t_lat: "center_lat"}, inplace=True)

    out["support_id"] = sup_near[s_id].to_numpy()
    out["support_lon"] = sup_near[s_lon].to_numpy()
    out["support_lat"] = sup_near[s_lat].to_numpy()
    out["support_dist2id_m"] = dist_m

    # If you want all original df_target columns, merge back:
    out_full = tar.reset_index(drop=True).copy()
    out_full["support_id"] = out["support_id"].to_numpy()
    out_full["support_lon"] = out["support_lon"].to_numpy()
    out_full["support_lat"] = out["support_lat"].to_numpy()
    out_full["support_dist2id_m"] = out["support_dist2id_m"].to_numpy()

    return out_full  # or return `out` if you only want the essentials



supporting_files = glob.glob(os.path.join(supporting_dir, '*.h5'))

df_target = pd.read_csv(displacement_file)

supporting_d = {}
supporting_d['event_id'] = []
supporting_d['lon'] = []
supporting_d['lat'] = []

for support_path in supporting_files:
    support_d = load_landslide_hdf5(support_path)
    
    support_event_id = support_path.split('/')[-1].split('-')[0]
    
    print(support_event_id)
    support_lat = support_d['meta']['lat']
    support_lon = support_d['meta']['lon']
    
    supporting_d['event_id'].append(support_event_id)
    supporting_d['lon'].append(support_lon)
    supporting_d['lat'].append(support_lat)
    
df_support = pd.DataFrame.from_dict(supporting_d)

df_map = map_target_to_support(df_support, df_target)

df_map.to_csv(os.path.join(ts_final_dir, "final_selection_only_mapped.csv"), index=False)
    
    



