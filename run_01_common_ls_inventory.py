#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:50:20 2025

@author: daniellelindsay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landslide inventory collation and enrichment script
Produces one CSV with polygon stats + per-source original IDs.
"""

import os
import subprocess
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
from pyproj import Geod
import numpy as np
from itertools import combinations
import fiona

from NC_Landslides_paths import common_paths

geod = Geod(ellps="WGS84")

# CRS definitions
SRC_CRS = 3310
DST_CRS = 4326

# Clustering thresholds
DIST_TH    = 500     # meters
OVERLAP_TH = 0.3     # fraction

# Bounding box
bbox_coords = [
    (-125.249, 42.389),
    (-121.018, 41.843),
    (-121.833, 38.672),
    (-125.864, 39.22),
    (-125.249, 42.389)
]
bbox_poly = Polygon(bbox_coords)

# Filters
area_min      = 5000
diameter_max  = 60000

# 1) Convert KMLs → GeoJSON
geojson_sources = {}
for src, kml in common_paths['ls_kmls'].items():
    geojson = common_paths['ls_geojson'][src]
    layers = fiona.listlayers(kml)

    # (same logic as before…) convert & merge multi-layer
    # … [omitted for brevity; your existing code goes here]
    # at the end:
    geojson_sources[src] = geojson

# 2) Load all polygons and tag by source
frames = []
for src, gj in geojson_sources.items():
    gdf = gpd.read_file(gj)
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf['source']   = src
    frames.append(gdf[['geometry','source']])
all_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

# 3) Build overlap graph
all_gdf = all_gdf.to_crs(epsg=SRC_CRS)
all_gdf['centroid'] = all_gdf.geometry.centroid
all_gdf['area']     = all_gdf.geometry.area
idx = all_gdf.sindex
G = nx.Graph()
for i, row in all_gdf.iterrows():
    G.add_node(i)
    minx, miny, maxx, maxy = row.geometry.bounds
    bounds = (minx-DIST_TH, miny-DIST_TH, maxx+DIST_TH, maxy+DIST_TH)
    for j in idx.intersection(bounds):
        if j <= i: continue
        if row.centroid.distance(all_gdf.at[j,'centroid'])>DIST_TH: continue
        inter = row.geometry.intersection(all_gdf.at[j,'geometry'])
        if inter.is_empty: continue
        if inter.area / min(row.area, all_gdf.at[j,'area']) >= OVERLAP_TH:
            G.add_edge(i,j)

# 4) Extract components → stats
records = []
for comp in nx.connected_components(G):
    subset    = all_gdf.loc[list(comp)]
    union_poly = unary_union(subset.geometry)
    if union_poly.is_empty: 
        continue

    # single largest polygon
    if union_poly.geom_type=='MultiPolygon':
        union_poly = max(union_poly.geoms, key=lambda p:p.area)

    # reproject to geographic for geodetic calcs
    geo_poly = gpd.GeoSeries([union_poly], crs=SRC_CRS).to_crs(epsg=DST_CRS).iloc[0]
    cent_geo = geo_poly.centroid

    # area & perimeter
    lonlat = list(geo_poly.exterior.coords)
    lons, lats = zip(*[(x,y) for x,y,*_ in lonlat])
    signed_area, perimeter = geod.polygon_area_perimeter(lons, lats)
    A = abs(signed_area)

    # compactness
    C = 4*np.pi*A/perimeter**2 if perimeter>0 else np.nan

    # min rotated rect
    mrr = geo_poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]
    (x0,y0),(x1,y1),(x2,y2) = coords[:3]
    _,_,d01 = geod.inv(x0,y0,x1,y1)
    _,_,d12 = geod.inv(x1,y1,x2,y2)
    if d01 >= d12:
        long_len, short_len = d01, d12
        orient = geod.inv(x0,y0,x1,y1)[0]
    else:
        long_len, short_len = d12, d01
        orient = geod.inv(x1,y1,x2,y2)[0]
    axis_ratio = long_len/short_len if short_len>0 else np.nan

    # convex hull max diameter
    hull = geo_poly.convex_hull
    pts = [(x,y) for x,y,*_ in hull.exterior.coords[:-1]]
    max_d = 0
    for (xa,ya),(xb,yb) in combinations(pts,2):
        _,_,d = geod.inv(xa,ya, xb,yb)
        if d>max_d: max_d=d

    records.append({
        'center_lat':      cent_geo.y,
        'center_lon':      cent_geo.x,
        'sources':         ",".join(sorted(subset['source'].unique())),
        'ls_area_m2':         A,
        'ls_perimeter_m':     perimeter,
        'ls_compactness':     C,
        'ls_max_diameter_m':  max_d,
        'ls_min_diameter_m':  short_len,
        'ls_axis_ratio':      axis_ratio,
        'ls_orientation_deg': orient,
    })

# build DataFrame & apply filters
df_stats = (pd
    .DataFrame(records)
    .pipe(lambda df: df[
        (df['ls_area_m2']      > area_min)   &
        (df['ls_max_diameter_m']< diameter_max)
    ])
    .reset_index(drop=True)
)
df_stats['ls_id'] = df_stats.index.map(lambda i: f"ls_{i+1:03d}")

# 5) Enrich: per-source original IDs
for src, gj in geojson_sources.items():
    gdf = gpd.read_file(gj).to_crs(epsg=DST_CRS)
    # pick ID field
    if   'FID'     in gdf: idf='FID'
    elif 'SlideID'in gdf: idf='SlideID'
    elif 'Name'   in gdf: idf='Name'
    else:               idf=None

    col = f"{src}_orig_id"
    df_stats[col] = ""  # initialize

    for idx, row in df_stats.iterrows():
        pt = Point(row['center_lon'], row['center_lat'])
        matches = gdf[gdf.contains(pt)]
        if not matches.empty:
            vals = matches[idf] if idf else matches.index
            df_stats.at[idx, col] = ";".join(map(str, vals))

# reorder columns for output
out_cols = [
    'ls_id','center_lat','center_lon','ls_area_m2','ls_perimeter_m','ls_compactness',
    'ls_max_diameter_m','ls_min_diameter_m','ls_axis_ratio','ls_orientation_deg',
    'sources'
] + sorted([c for c in df_stats if c.endswith('_orig_id')])

# 6) Save a single enriched CSV
out_path = common_paths['ls_inventory']
df_stats.to_csv(out_path, index=False, columns=out_cols)
print(f"Enriched inventory saved to {out_path}")
