#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landslide inventory collation and enrichment script

"""

import os
import subprocess
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import Point
import fiona
from shapely.geometry import Point, Polygon
from pyproj import Geod
import numpy as np
from itertools import combinations

from NC_Landslides_paths import common_paths

geod = Geod(ellps="WGS84")

# Coordinate reference systems
SRC_CRS = 3310                   # projected CRS for clustering
DST_CRS = 4326                   # geographic CRS for geodetic calcs

# Graph clustering thresholds
DIST_TH = 500        # meters: max centroids distance
OVERLAP_TH = 0.3     # fraction: min overlap area

# Bounding box for final inventory (lon, lat)
bbox_coords = [
    (-125.249, 42.389),
    (-121.018, 41.843),
    (-121.833, 38.672),
    (-125.864, 39.22),
    (-125.249, 42.389)
]

# --- 1. Convert all KML inputs to GeoJSON (handles multi-layer KMLs) ---
geojson_sources = {}
for src, kml_file in common_paths['ls_kmls'].items():
    geojson_file = common_paths['ls_geojson'][src]
    layers = fiona.listlayers(kml_file)
    if len(layers) <= 1:
        # single-layer: direct convert if missing
        if not os.path.exists(geojson_file):
            os.makedirs(os.path.dirname(geojson_file), exist_ok=True)
            subprocess.run([
                'ogr2ogr', '-skipfailures', '-f', 'GeoJSON', geojson_file, kml_file
            ], check=True)
    else:
        # multi-layer: convert each layer then merge
        parts = []
        for layer in layers:
            temp_geo = os.path.splitext(geojson_file)[0] + f"__{layer}.geojson"
            if not os.path.exists(temp_geo):
                os.makedirs(os.path.dirname(temp_geo), exist_ok=True)
                subprocess.run([
                    'ogr2ogr', '-skipfailures', '-f', 'GeoJSON', temp_geo, kml_file, layer
                ], check=True)
            try:
                parts.append(gpd.read_file(temp_geo))
            except Exception:
                pass
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=parts[0].crs)
            merged.to_file(geojson_file, driver='GeoJSON')
        # clean up temp files
        for layer in layers:
            tmp = os.path.splitext(geojson_file)[0] + f"__{layer}.geojson"
            if os.path.exists(tmp):
                os.remove(tmp)
    geojson_sources[src] = geojson_file

# --- Convert all KML inputs to GMT text via GMT’s kml2gmt ---
gmt_sources = {}
for src, kml_file in common_paths['ls_kmls'].items():
   gmt_file = common_paths['ls_gmt'][src]
   if not os.path.exists(gmt_file):
       os.makedirs(os.path.dirname(gmt_file), exist_ok=True)
       with open(gmt_file, 'w') as out:
           subprocess.run(
               ['gmt', 'kml2gmt', kml_file],
               stdout=out,
               check=True
           )
   gmt_sources[src] = gmt_file


# --- 2. Load and tag each polygon by source ---
frames = []
for src, path in geojson_sources.items():
    gdf = gpd.read_file(path)
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf['source'] = src
    frames.append(gdf[['geometry', 'source']])
all_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

# --- 3. Build spatial graph of overlapping polygons ---
all_gdf = all_gdf.to_crs(epsg=SRC_CRS)
all_gdf['centroid'] = all_gdf.geometry.centroid
all_gdf['area'] = all_gdf.geometry.area
idx = all_gdf.sindex
graph = nx.Graph()

for i, row in all_gdf.iterrows():
    graph.add_node(i)
    minx, miny, maxx, maxy = row.geometry.bounds
    candidate_bounds = (minx - DIST_TH, miny - DIST_TH, maxx + DIST_TH, maxy + DIST_TH)
    for j in idx.intersection(candidate_bounds):
        if j <= i: continue
        if row.centroid.distance(all_gdf.at[j, 'centroid']) > DIST_TH: continue
        inter = row.geometry.intersection(all_gdf.at[j, 'geometry'])
        if inter.is_empty: continue
        if inter.area / min(row.area, all_gdf.at[j, 'area']) >= OVERLAP_TH:
            graph.add_edge(i, j)

# --- 4. Extract connected components as unique features ---
groups = nx.connected_components(graph)
records = []
for grp in groups:
    subset     = all_gdf.loc[list(grp)]
    union_poly = unary_union(subset.geometry)
    if union_poly.is_empty:
        continue

    # collect the list of original source keys
    sources = ",".join(sorted(subset['source'].unique()))

    # --- new: ensure a single Polygon (pick largest part if Multi) ---
    if union_poly.geom_type == 'MultiPolygon':
        # pick the largest member polygon by area
        union_poly = max(union_poly.geoms, key=lambda p: p.area)

    # 1) centroid in WGS84
    cent    = union_poly.centroid
    wgs_pt  = gpd.GeoSeries([cent], crs=SRC_CRS).to_crs(epsg=DST_CRS).iloc[0]
    geo_poly = gpd.GeoSeries([union_poly], crs=SRC_CRS).to_crs(epsg=DST_CRS).iloc[0]

    # 2) geodetic area & perimeter
    raw_coords = list(geo_poly.exterior.coords)
    lonlat     = [(lon, lat) for lon, lat, *rest in raw_coords]
    lons, lats = zip(*lonlat)
    signed_area, perimeter_m = geod.polygon_area_perimeter(lons, lats)
    area_m2                  = abs(signed_area)

    # 3) compactness
    compactness = 4 * np.pi * area_m2 / perimeter_m**2 if perimeter_m > 0 else np.nan

    # 4) minimum‐rotated rectangle
    mrr_geo     = geo_poly.minimum_rotated_rectangle
    rc          = list(mrr_geo.exterior.coords)[:-1]
    (x0,y0),(x1,y1),(x2,y2) = rc[:3]
    _,_,d01 = geod.inv(x0,y0, x1,y1)
    _,_,d12 = geod.inv(x1,y1, x2,y2)
    if d01>=d12:
        long_len, short_len = d01,d12
        orientation_deg     = geod.inv(x0,y0, x1,y1)[0]
    else:
        long_len, short_len = d12,d01
        orientation_deg     = geod.inv(x1,y1, x2,y2)[0]
    axis_ratio    = long_len/short_len if short_len>0 else np.nan
    min_diameter  = short_len

    # 5) convex‐hull maximum diameter
    hull_geo      = geo_poly.convex_hull
    pts           = [(lon, lat) for lon, lat, *rest in hull_geo.exterior.coords[:-1]]
    max_diameter  = 0.0
    for (xA,yA),(xB,yB) in combinations(pts, 2):
        _,_,dAB = geod.inv(xA,yA, xB,yB)
        if dAB>max_diameter:
            max_diameter = dAB

    # record all seven metrics…
    records.append({
      'center_lat':      wgs_pt.y,
      'center_lon':      wgs_pt.x,
      'sources':         sources, 
      'area_m2':         area_m2,
      'perimeter_m':     perimeter_m,
      'compactness':     compactness,
      'max_diameter_m':  max_diameter,
      'min_diameter_m':  min_diameter,
      'axis_ratio':      axis_ratio,
      'orientation_deg': orientation_deg,
    })
    


# --- 5. Build DataFrame, assign IDs & filter by bbox ---
df = pd.DataFrame(records)
df = df.sort_values('center_lat').reset_index(drop=True)

# --- 5a. Apply geographic bounding box filter for frame ALOS2433032800-220530 ---
bbox_poly = Polygon(bbox_coords)
df = df[df.apply(lambda r: bbox_poly.contains(Point(r['center_lon'], r['center_lat'])), axis=1)].reset_index(drop=True)

df['ls_id'] = df.index.map(lambda i: f"ls_{i+1:03d}")
df = df[['ls_id','center_lat','center_lon',
         'area_m2','perimeter_m','compactness',
         'max_diameter_m','min_diameter_m',
         'axis_ratio','orientation_deg', 'sources',]]

# --- 6. Save base inventory ---
out = common_paths['ls_inventory_stats']
df.to_csv(out, index=False)
print(f"Base inventory saved to {out}")

# --- 7. Enrich with per-source original IDs ---
inv_df = pd.read_csv(out)
for src, path in geojson_sources.items():
    gdf = gpd.read_file(path).to_crs(epsg=DST_CRS)
    if 'FID' in gdf.columns:
        idf = 'FID'
    elif 'SlideID' in gdf.columns:
        idf = 'SlideID'
    elif 'Name' in gdf.columns:
        idf = 'Name'
    else:
        idf = None
    gdf['orig_id'] = gdf[idf] if idf else gdf.index.astype(str)
    inv_df[src + '_orig_id'] = ''
    for i, row in inv_df.iterrows():
        pt = Point(row['center_lon'], row['center_lat'])
        matches = gdf[gdf.contains(pt)]
        if not matches.empty:
            inv_df.at[i, src + '_orig_id'] = ';'.join(matches['orig_id'].dropna().astype(str))
# --- 8. Save enriched inventory ---
enriched = common_paths['ls_inventory']
inv_df.to_csv(enriched, index=False)
print(f"Enriched inventory saved to {enriched}")
