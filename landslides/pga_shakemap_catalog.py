#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 21:21:36 2025

@author: andrewmcnab
"""

import requests
import io
import h5py
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger("my_app")

class ShakeMapError(Exception):
    """Custom exception for ShakeMap retrieval/conversion errors."""

def fetch_usgs_events(starttime: str,
                      endtime: str,
                      latitude: float,
                      longitude: float,
                      maxradiuskm: float):
    """
    Query USGS for all events with ShakeMap products between starttime and endtime
    within maxradiuskm of (latitude, longitude).
    Returns the GeoJSON 'features' list.
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format":       "geojson",
        "starttime":    starttime,
        "endtime":      endtime,
        "latitude":     latitude,
        "longitude":    longitude,
        "maxradiuskm":  maxradiuskm,
        "producttype":  "shakemap"
    }
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    return r.json()["features"]

def get_shakemap_df(event_id: str,
                    locations,
                    imts=('PGA', 'PGV', 'MMI')):

    try:
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={"eventid": event_id, "format": "geojson", "producttype": "shakemap"},
            timeout=60
        )
        resp.raise_for_status()
        prods = resp.json()["properties"]["products"]["shakemap"]
        latest = max(prods, key=lambda p: p["updateTime"])
        hdf_url = latest["contents"]["download/shake_result.hdf"]["url"]
    except Exception as e:
        raise ShakeMapError(f"Error discovering ShakeMap HDF URL for {event_id}: {e}")
        
    try:
        bio = io.BytesIO(requests.get(hdf_url, timeout=60).content)
    except Exception as e:
        raise ShakeMapError(f"Error downloading HDF5 for {event_id}: {e}")

    try:
        with h5py.File(bio, "r") as f:
            base0 = f"/arrays/imts/GREATER_OF_TWO_HORIZONTAL/{imts[0]}/mean"
            attrs = f[base0].attrs
            nx, ny = int(attrs["nx"]), int(attrs["ny"])
            xmin, ymax = float(attrs["xmin"]), float(attrs["ymax"])
            dx, dy = float(attrs["dx"]), float(attrs["dy"])

            ds = {}
            for imt in imts:
                grp = f"/arrays/imts/GREATER_OF_TWO_HORIZONTAL/{imt}"
                ds[imt] = {
                    "mean": f[f"{grp}/mean"],
                    "std":  f[f"{grp}/std"],
                    "u_mean": f[f"{grp}/mean"].attrs.get("units",""),
                    "u_std":  f[f"{grp}/std"].attrs.get("units","")
                }
                
            recs = []
            for i, (lat, lon) in enumerate(locations):
                col = int(round((lon - xmin)/dx))
                row = int(round((ymax - lat)/dy))
                if not (0 <= col < nx and 0 <= row < ny):
                    logger.warning(f"{event_id}: point {lat},{lon} out of grid bounds; skipping site index {i}")
                    continue
                rowrec = {"idx": i, "lat": lat, "lon": lon}
                for imt, H in ds.items():
                    raw_m = H["mean"][row, col]
                    raw_s = H["std"][row, col]
                    mean = float(np.exp(raw_m)) if "ln" in H["u_mean"].lower() else float(raw_m)
                    std  = float(mean * raw_s)    if "ln" in H["u_std"].lower()  else float(raw_s)
                    rowrec[f"{imt.lower()}_mean"] = mean
                    rowrec[f"{imt.lower()}_std"]  = std
                recs.append(rowrec)
            
            if not recs:
                raise ShakeMapError(f"No in-bounds locations for {event_id}")
            
            df = pd.DataFrame.from_records(recs)
            return df

    except Exception as e:
        raise ShakeMapError(f"Error processing ShakeMap HDF for {event_id}: {e}")


def multi_site_shakemap_summary(sites, starttime, endtime, maxradiuskm):
    site_events = {}
    for site in sites:
        feats = fetch_usgs_events(starttime, endtime, site["lat"], site["lon"], maxradiuskm)
        site_events[site["site_id"]] = {
            "events":   [f["id"] for f in feats],
            "features": {f["id"]: f for f in feats}
        }

    event_to_sites = defaultdict(list)
    for sid, info in site_events.items():
        for eid in info["events"]:
            event_to_sites[eid].append(sid)

    logger.info(f"Processing {len(event_to_sites)} unique events")
    out_rows = []

    for event_id, site_ids in event_to_sites.items():
        coords = [(s["lat"], s["lon"]) for s in sites if s["site_id"] in site_ids]

        try:
            df_vals = get_shakemap_df(event_id, coords)
        except ShakeMapError as e:
            logger.error(f"Skipping event {event_id}: {e}")
            continue

        if "idx" not in df_vals.columns:
            logger.warning(f"{event_id}: 'idx' not in get_shakemap_df output; falling back to positional alignment")
            df_vals = df_vals.reset_index(drop=True)
            indexer = {i: i for i in range(len(df_vals))}
        else:
            df_vals = df_vals.set_index("idx")
            indexer = {i: i for i in df_vals.index}

        for i, sid in enumerate(site_ids):
            if i not in df_vals.index:
                logger.warning(f"Event {event_id}: site {sid} out of grid; skipping this site")
                continue

            try:
                feat = site_events[sid]["features"][event_id]
                evt_time = pd.to_datetime(feat["properties"]["time"], unit="ms", utc=True)
                row = df_vals.loc[i].to_dict()
                row.update({
                    "site_id":    sid,
                    "event_id":   event_id,
                    "event_time": evt_time
                })
                out_rows.append(row)
            except Exception as e:
                logger.error(f"Error building row for event {event_id}, site {sid}: {e}")

    df_out = pd.DataFrame(out_rows)

    col_order = [
        "site_id", "event_id", "event_time", "lat", "lon",
        "pga_mean", "pga_std", "pgv_mean", "pgv_std", "mmi_mean", "mmi_std"
    ]
    existing = [c for c in col_order if c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in existing]
    return df_out[existing + remaining] if not df_out.empty else df_out

if __name__ == "__main__":
    sites = [{'site_id': 'wc112', 'lon': -124.1197522215, 'lat': 40.4578218193},
     {'site_id': 'wc043', 'lon': -123.6561695485, 'lat': 40.47140339045},
     {'site_id': 'wc020', 'lon': -123.820257756, 'lat': 40.4726937151},
     {'site_id': 'wc348', 'lon': -123.693210421, 'lat': 40.49660514085},
     {'site_id': 'wc108', 'lon': -123.666015129, 'lat': 40.51103477565},
     {'site_id': 'wc109b', 'lon': -123.78534196, 'lat': 40.51254952515},
     {'site_id': 'wc109', 'lon': -123.79034196, 'lat': 40.51654952515},
     {'site_id': 'wc337', 'lon': -123.7219497605, 'lat': 40.6341669627},
     {'site_id': 'wc338', 'lon': -123.706086899, 'lat': 40.64310279015},
     {'site_id': 'wc044', 'lon': -123.7897893285, 'lat': 40.6575320207},
     {'site_id': 'wc345', 'lon': -123.521274308, 'lat': 40.66875593425},
     {'site_id': 'wc004', 'lon': -123.776274923, 'lat': 40.83348777405},
     {'site_id': 'wc411', 'lon': -123.9926834385, 'lat': 40.9287440053}]
    df = multi_site_shakemap_summary(
        sites,
        "2021-01-10", "2024-03-18",#"2024-03-18"
        maxradiuskm=100
    )
    print(df)