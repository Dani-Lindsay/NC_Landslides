#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 11:09:04 2025

@author: andrewmcnab
"""

import pandas as pd
import logging
import os
from datetime import datetime

from rainfall_prism import prism_daily_rainfall
from geology_macrostrat_usgs import get_geology
from pga_shakemap_catalog import multi_site_shakemap_summary
from geometry_tools import distance_to_nearest_road, distance_to_ocean, get_elevation
from hdf5_support import save_landslide_supporting
from project_logging import setup_logger

meta_data = '/Users/andrewmcnab/Data/2025/ls_analysis_v2/landslide_meta/final_selection.csv'

out_dir = '/Users/andrewmcnab/Data/2025/ls_analysis_v2/landslide_supporting'
log_dir = '/Users/andrewmcnab/Data/2025/ls_analysis_v2/logs'

time_now = str(datetime.now()).replace(' ', '_').replace(':','-')

log_name = f'sup_extr_{time_now}.log'

logger = setup_logger(log_dir, log_name)

eq_event_radius = 300
global_end_date = '2024-10-01'
global_start_date = '2020-10-01' #'2021-01-11'

df = pd.read_csv(meta_data)

sites = [{'site_id':row['ls_id'], 'lon': row['center_lon'], 'lat': row['center_lat']} for _,row in df.iterrows()]

landslide_supporting = {}

for site in sites:
    
    site_id = site['site_id']
    lon = site['lon']
    lat = site['lat']
    
    logger.info(f'running {site_id}')
    
    landslide_supporting[site_id] = {}
    
    landslide_supporting[site_id]['meta'] = {}
    landslide_supporting[site_id]['meta']['lon'] = lon
    landslide_supporting[site_id]['meta']['lat'] = lat
    landslide_supporting[site_id]['meta']['lat'] = lat
    landslide_supporting[site_id]['meta']['global_start_date'] = global_start_date
    landslide_supporting[site_id]['meta']['global_end_date'] = global_end_date
    landslide_supporting[site_id]['meta']['eq_event_radius'] = eq_event_radius
    
    try:
        rainfall_df = prism_daily_rainfall(lat, lon, global_start_date, global_end_date)
        rainfall_d = rainfall_df.to_dict(orient='list') 
    except Exception as e:
        logger.warning(f'{site_id} rainfall extract failed, setting to None: {e}')
        rainfall_d = None
        
    landslide_supporting[site_id]["rainfall"] = rainfall_d
    
    try:
        pt = get_geology(lat, lon)
    except Exception as e:
        logger.warning(f'{site_id} geology extract failed, setting to None: {e}')
        pt = None
    
    landslide_supporting[site_id]["geology"] = pt
    
    try:
        d_road = distance_to_nearest_road(lat, lon, search_radius=3000)
    except Exception as e:
        logger.warning(f'{site_id} nearest road extract failed, setting to None: {e}')
        d_road = None
        
    try:
        d_ocean = distance_to_ocean(lat, lon)
    except Exception as e:
        logger.warning(f'{site_id} nearest ocean extract failed, setting to None: {e}')
        d_ocean = None
    
    try:
        r_elevation = get_elevation(lat, lon)
    except Exception as e:
        logger.warning(f'{site_id} elevation extract failed, setting to None: {e}')
        r_elevation = None
    
    landslide_supporting[site_id]["distances"] = {}
    landslide_supporting[site_id]["distances"]["road_m"] = d_road
    landslide_supporting[site_id]["distances"]["ocean_m"] = d_ocean
    landslide_supporting[site_id]["distances"]["elevation_m"] = r_elevation

sitenames = landslide_supporting.keys()
sites_torun = [entry for entry in sites if entry['site_id'] in sitenames]

pga_df = multi_site_shakemap_summary(
    sites_torun,
    global_start_date, global_end_date,
    maxradiuskm=eq_event_radius
)

for site_id in sitenames:
    
    
    
    pga_ = pga_df[pga_df.site_id==site_id]
    
    if pga_.empty:
        logger.warning(f'no events found for {site_id}')
        landslide_supporting[site_id]["pga"] = {}
        continue
    
    pga_d = pga_.to_dict(orient='list') 
    landslide_supporting[site_id]["pga"] = pga_d  


save_landslide_supporting(landslide_supporting, out_dir)



